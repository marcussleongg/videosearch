#!/usr/bin/env python3
"""
Ingest a video into the search index.

Usage:
  python ingest.py video.mp4              # single video
  python ingest.py --folder /path/to/dir  # all .mp4s in a folder
  python ingest.py video.mp4 --dry-run    # test without writing to DBs
"""

import argparse
import re
import sys
from pathlib import Path
from uuid import uuid4

import config
from embeddings import extract_video_frames, get_description, get_embedding


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


def ingest(video_path: str, dry_run: bool) -> bool:
    """Returns True on success, False on failure."""
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        print(f"File not found: {path}")
        return False

    print(f"Video:  {path.name}")

    print("Extracting frames (1 per 2s)...")
    frames, duration = extract_video_frames(str(path))
    print(f"  {len(frames)} frames extracted  |  duration: {duration:.1f}s")

    print("Generating description with Gemma4...")
    description = get_description(frames)

    angle = re.search(r'\[Angle: (.+?)\]', description)
    footage = re.search(r'\[Footage: (.+?)\]', description)
    angle = angle.group(1) if angle else None
    footage = footage.group(1) if footage else None

    print(f"\nDescription:\n{description}")
    print(f"Angle: {angle}  |  Footage: {footage}\n")

    if dry_run:
        print("[dry-run] Skipping Pinecone and Supabase writes.")
        return True

    print("Embedding description...")
    vector = get_embedding(description)
    print(f"  Vector dim: {len(vector)}")

    video_id = str(uuid4())

    print("Inserting to Supabase...")
    client = get_supabase_client()
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
    print(f"  ID: {video_id}")

    print("Upserting to Pinecone...")
    index = get_pinecone_index()
    index.upsert(vectors=[{
        "id": video_id,
        "values": vector,
        "metadata": {
            "filename": path.name,
            "description": description[:1000],
            "duration_s": duration,
            "frame_count": len(frames),
            "angle": angle,
            "footage": footage,
        },
    }])
    print(f"  Pinecone ID: {video_id}")

    print("Done.")
    return True


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
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            sys.exit(f"No .mp4 files found in {folder}")
        print(f"Found {len(videos)} video(s) in {folder}\n")
        succeeded, failed = [], []
        for i, video in enumerate(videos, 1):
            print(f"[{i}/{len(videos)}] {video.name}")
            print("-" * 40)
            ok = ingest(str(video), args.dry_run)
            (succeeded if ok else failed).append(video.name)
            print()
        print("=" * 40)
        print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
        if failed:
            print("Failed:")
            for name in failed:
                print(f"  - {name}")
    else:
        ingest(args.video, args.dry_run)


if __name__ == "__main__":
    main()
