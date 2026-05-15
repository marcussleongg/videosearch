#!/usr/bin/env python3
"""
Ingest a video into the search index.

Usage:
  python ingest.py video.mp4              # single video
  python ingest.py --folder /path/to/dir  # all .mp4s in a folder
  python ingest.py video.mp4 --dry-run    # test without writing to DBs
"""

import argparse
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import config
from embeddings import get_description, get_embedding


def extract_video_frames(video_path: str) -> tuple[list[str], float]:
    """Return (list of base64 JPEG frames, duration in seconds)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-of", "csv=p=0",
         "-show_entries", "format=duration", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 60.0

    num_frames = max(1, int(duration / 2))
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_frames):
            t = duration * (i + 0.5) / num_frames
            out = f"{tmpdir}/frame{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", str(t), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", out],
                capture_output=True,
            )
            p = Path(out)
            if p.exists():
                frames.append(base64.b64encode(p.read_bytes()).decode())

    return frames, duration


def get_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(config.PINECONE_INDEX)


def get_supabase_client():
    from supabase import create_client
    return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def already_ingested(client, filename: str) -> bool:
    result = client.table("videos").select("id").eq("filename", filename).execute()
    return len(result.data) > 0


def ingest(video_path: str, dry_run: bool, client=None, index=None) -> str:
    """Returns 'ok', 'skipped', or 'error'."""
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        print(f"  File not found: {path}")
        return "error"

    try:
        if not dry_run:
            if client is None:
                client = get_supabase_client()
            if already_ingested(client, path.name):
                print(f"  Skipping — already ingested.")
                return "skipped"

        print("  Extracting frames (1 per 2s)...")
        frames, duration = extract_video_frames(str(path))
        print(f"  {len(frames)} frames extracted  |  duration: {duration:.1f}s")

        print("  Generating description with Gemma4...")
        description = get_description(frames)

        angle = re.search(r'\[Angle: (.+?)\]', description)
        footage = re.search(r'\[Footage: (.+?)\]', description)
        angle = angle.group(1) if angle else None
        footage = footage.group(1) if footage else None

        print(f"\n  Description:\n  {description}")
        print(f"  Angle: {angle}  |  Footage: {footage}\n")

        if dry_run:
            print("  [dry-run] Skipping Pinecone and Supabase writes.")
            return "ok"

        print("  Embedding description...")
        vector = get_embedding(description)

        video_id = str(uuid4())

        print("  Inserting to Supabase...")
        client.table("videos").insert({
            "id": video_id,
            "filename": path.name,
            "file_path": str(path),
            "description": description,
            "duration_s": duration,
            "frame_count": len(frames),
            "angle": angle,
            "footage": footage,
        }).execute()

        print("  Upserting to Pinecone...")
        if index is None:
            index = get_pinecone_index()
        index.upsert(vectors=[{
            "id": video_id,
            "values": vector,
            "metadata": {
                "filename": path.name,
                "description": description[:1000],
                "duration_s": duration,
                "frame_count": len(frames),
                "angle": angle or "unknown",
                "footage": footage or "unknown",
            },
        }])

        print(f"  Done. ID: {video_id}")
        return "ok"

    except Exception as e:
        print(f"  ERROR: {e}")
        return "error"


def main():
    parser = argparse.ArgumentParser(description="Ingest a video into the search index.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("video", nargs="?", help="Path to a single video file")
    group.add_argument("--folder", help="Path to a folder of .mp4 files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print description only, skip DB writes")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder).expanduser().resolve()
        videos = sorted(p for p in folder.glob("*.mp4") if not p.name.startswith("._"))
        if not videos:
            sys.exit(f"No .mp4 files found in {folder}")
        print(f"Found {len(videos)} video(s) in {folder}\n")
        client = None if args.dry_run else get_supabase_client()
        index = None if args.dry_run else get_pinecone_index()
        succeeded, skipped, failed = [], [], []
        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {video.name}")
            print("-" * 40)
            status = ingest(str(video), args.dry_run, client=client, index=index)
            if status == "ok":
                succeeded.append(video.name)
            elif status == "skipped":
                skipped.append(video.name)
            else:
                failed.append(video.name)
            print()
        print("=" * 40)
        print(f"Done: {len(succeeded)} ingested, {len(skipped)} skipped, {len(failed)} failed")
        if failed:
            print("Failed:")
            for name in failed:
                print(f"  - {name}")
    else:
        status = ingest(args.video, args.dry_run)
        if status == "skipped":
            print("Already ingested, skipping.")
        elif status == "error":
            sys.exit(1)


if __name__ == "__main__":
    main()
