#!/usr/bin/env python3
"""
Polite-Guard Evaluation Script
- Loads JSON/JSONL answer files
- Uses HuggingFace `Intel/polite-guard` for politeness classification
- Maps scores to a continuous range [-1, 1]
- Supports batch processing and CSV output
"""

import os, io, glob, json, csv
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import pipeline


def read_records(path: str) -> List[Dict[str, Any]]:
    """Read a JSON/JSONL file into a list of dict records."""
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # Try JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Support dict-of-lists format
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return []
    except Exception:
        pass
    # Fallback: JSONL format
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except:
            continue
    return recs


def polite_guard_score(pipe, text: str) -> float:
    """Run Polite-Guard model on a single text and map score to [-1, 1]."""
    out = pipe(text, truncation=True)[0]
    label = (out.get("label") or "").lower()
    s = float(out.get("score", 0.0))
    p_polite = s if label == "polite" else (1.0 - s)
    return 2.0 * p_polite - 1.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder of *.json input files (answers)")
    ap.add_argument("--out", default="scores_polite_guard.csv", help="Output CSV file")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for model inference")
    ap.add_argument("--dry_run", type=int, default=0, help="Debug mode: only process the first K samples")
    args = ap.parse_args()

    # Collect all records from JSON files
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    items = []
    for fp in files:
        for i, r in enumerate(read_records(fp)):
            items.append({
                "file": os.path.basename(fp),
                "model_in_file": r.get("model", ""),
                "query_id": r.get("query_id", ""),
                "answer": str(r.get("answer", "")).replace("\n", " ")
            })
    if args.dry_run and len(items) > args.dry_run:
        items = items[:args.dry_run]

    print(f"Loaded {len(items)} answers from {len(files)} files.")

    # Initialize HuggingFace pipeline
    pipe = pipeline("text-classification", model="Intel/polite-guard", device_map="auto", truncation=True)

    # Process in batches and write to CSV
    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "model_in_file", "query_id", "polite_guard_score"])

        bs = max(1, args.batch_size)
        for start in tqdm(range(0, len(items), bs), desc="polite-guard"):
            batch = items[start:start+bs]
            texts = [x["answer"] for x in batch]
            outputs = pipe(texts, truncation=True)
            # One output per text
            for rec, out in zip(batch, outputs):
                label = (out.get("label") or "").lower()
                s = float(out.get("score", 0.0))
                p_polite = s if label == "polite" else (1.0 - s)
                score = 2.0 * p_polite - 1.0
                w.writerow([rec["file"], rec["model_in_file"], rec["query_id"], score])

    print(f"Done. Wrote -> {args.out}")


if __name__ == "__main__":
    main()
