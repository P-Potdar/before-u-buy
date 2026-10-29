#!/usr/bin/env python3
"""Download a bounded, traceable sample of a permitted public dataset.

This streams the official Hugging Face file and stops after ``--max-records``;
it does not download the entire 146 MB source for a pilot. The retained rows
are local research data with source attribution and are not customer records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.public_data import MANIFEST_PATH, PUBLIC_DATA_DIR, RAW_REVIEWS_PATH, normalize_text  # noqa: E402

SOURCE_URL = "https://huggingface.co/datasets/stephaniestv/Electronics_Product_Review_With_Sentiment/resolve/main/amazon_electronics_review_sentiment.csv?download=true"
SOURCE_PAGE = "https://huggingface.co/datasets/stephaniestv/Electronics_Product_Review_With_Sentiment"
SOURCE_COMMIT = "778bd6e"
LICENSE = "Apache-2.0 (dataset repository); underlying Amazon/McAuley source terms require separate review"
CACHE_PATH = PUBLIC_DATA_DIR / "amazon_electronics_review_sentiment.csv"


def resumable_download(destination: Path, max_bytes: int) -> None:
    """Download the source with HTTP Range continuation into a local cache."""
    partial = destination.with_suffix(destination.suffix + ".part")
    while True:
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "BeforeYouBuyAI/2.0 public-data-pilot"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(SOURCE_URL, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", response.getcode())
            if start and status != 206:
                partial.unlink(missing_ok=True)
                continue
            mode = "ab" if start else "wb"
            with partial.open(mode) as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    if output.tell() > max_bytes:
                        raise RuntimeError(f"source cache exceeded --max-bytes={max_bytes}")
            content_range = response.headers.get("Content-Range", "")
            total_match = re.search(r"/([0-9]+)$", content_range)
            total = int(total_match.group(1)) if total_match else int(response.headers.get("Content-Length", "0")) + start
        if total and partial.stat().st_size >= total:
            partial.replace(destination)
            return
        if not total:
            partial.replace(destination)
            return


def download(max_records: int, max_bytes: int, cache_source: bool) -> None:
    PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    source_file = None
    if cache_source:
        resumable_download(CACHE_PATH, max_bytes)
        source_file = CACHE_PATH.open("rb")
    temporary = RAW_REVIEWS_PATH.with_suffix(".jsonl.part")
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "BeforeYouBuyAI/2.0 public-data-pilot"})
    retained = rejected = bytes_seen = scanned = 0
    digest = hashlib.sha256()
    buckets: dict[str, list[dict[str, object]]] = {}
    rng = random.Random(20261028)
    last_error = None
    for attempt in range(3):
        try:
            source = source_file or urllib.request.urlopen(request, timeout=30)
            with source as response, temporary.open("w", encoding="utf-8") as output:
                reader = csv.DictReader((line.decode("utf-8", errors="replace") for line in response))
                for row in reader:
                    scanned += 1
                    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                    bytes_seen += len(encoded)
                    if bytes_seen > max_bytes:
                        raise RuntimeError(f"download sample exceeded --max-bytes={max_bytes}")
                    text = (row.get("text") or "").strip()
                    if not text:
                        rejected += 1
                        continue
                    clean = {
                        "source_record_id": f"{row.get('parent_asin') or row.get('asin') or 'unknown'}:{row.get('timestamp') or scanned}:{scanned}",
                        "title": row.get("title") or "",
                        "text": text,
                        "rating": row.get("rating"),
                        "asin": row.get("asin") or None,
                        "parent_asin": row.get("parent_asin") or None,
                        "timestamp": row.get("timestamp") or None,
                        "helpful_vote": row.get("helpful_vote") or row.get("helpful_votes") or None,
                        "verified_purchase": row.get("verified_purchase") or None,
                        "source_sentiment": row.get("review_sentiment") or "Unknown",
                        "source_url": SOURCE_PAGE,
                        "license": LICENSE,
                    }
                    label = str(clean["source_sentiment"])
                    bucket = buckets.setdefault(label, [])
                    # Deterministic reservoir sampling gives the pilot coverage
                    # even when the source CSV is grouped by sentiment.
                    target = max(1, max_records // 3)
                    if len(bucket) < target:
                        bucket.append(clean)
                    else:
                        replacement = rng.randrange(scanned)
                        if replacement < target:
                            bucket[replacement] = clean
                selected = []
                for label in sorted(buckets):
                    selected.extend(buckets[label])
                selected = selected[:max_records]
                seen_text: set[str] = set()
                for clean in selected:
                    fingerprint = normalize_text(f"{clean['title']} {clean['text']}")
                    if fingerprint in seen_text:
                        rejected += 1
                        continue
                    seen_text.add(fingerprint)
                    output.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    retained += 1
                    digest.update(json.dumps(clean, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            temporary.replace(RAW_REVIEWS_PATH)
            break
        except Exception as exc:  # bounded retry; no fabricated fallback
            last_error = exc
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    manifest = {
        "source_name": "stephaniestv/Electronics_Product_Review_With_Sentiment",
        "source_url": SOURCE_PAGE,
        "download_url": SOURCE_URL,
        "dataset_commit": SOURCE_COMMIT,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "declared_license": LICENSE,
        "original_source_attribution": "Dataset card attributes raw data to the Amazon Product Review dataset by McAuley Lab.",
        "download_method": "streamed official Hugging Face CSV; bounded retained sample",
        "resumable_source_cache": str(CACHE_PATH) if cache_source else None,
        "download_sha256_of_retained_source_rows": digest.hexdigest(),
        "records_retained": retained,
        "records_rejected_missing_text": rejected,
        "records_scanned": scanned,
        "source_sentiment_buckets": {label: len(rows) for label, rows in sorted(buckets.items())},
        "max_records": max_records,
        "max_bytes_accounted": max_bytes,
        "provenance_limitations": [
            "The repository declares Apache-2.0, but the underlying Amazon review material may have additional terms.",
            "The source page says review_sentiment was classified with Mistral Large Language Model; it is retained as a source annotation, not independently verified truth.",
            "No reviewer identifiers are retained.",
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(RAW_REVIEWS_PATH), "records_retained": retained, "rejected": rejected, "manifest": str(MANIFEST_PATH)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["electronics_reviews"], default="electronics_reviews")
    parser.add_argument("--max-records", type=int, default=5000)
    parser.add_argument("--max-bytes", type=int, default=220_000_000)
    parser.add_argument("--cache-source", action="store_true", help="keep a resumable local copy of the source CSV")
    args = parser.parse_args()
    if args.max_records < 1 or args.max_records > 100_000:
        raise SystemExit("--max-records must be between 1 and 100000")
    download(args.max_records, args.max_bytes, args.cache_source)


if __name__ == "__main__":
    main()
