#!/usr/bin/env python3
import os, io, glob, json, csv
from typing import List, Dict, Any
from tqdm import tqdm

from transformers import pipeline

def read_records(path: str) -> List[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v
            return []
    except Exception:
        pass
    # JSONL
    recs = []
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        try:
            recs.append(json.loads(line))
        except: 
            continue
    return recs

def polite_guard_score(pipe, text: str) -> float:
    # 截断避免超长
    out = pipe(text, truncation=True)[0]
    label = (out.get("label") or "").lower()
    s = float(out.get("score", 0.0))
    p_polite = s if label == "polite" else (1.0 - s)
    # 映射到 [-1, 1]
    return 2.0 * p_polite - 1.0

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="folder of *.json (原始答案)")
    ap.add_argument("--out", default="scores_polite_guard.csv")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dry_run", type=int, default=0, help="仅前K条，调试用")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
    items = []
    for fp in files:
        for i, r in enumerate(read_records(fp)):
            items.append({
                "file": os.path.basename(fp),
                "model_in_file": r.get("model",""),
                "query_id": r.get("query_id",""),
                "answer": str(r.get("answer","")).replace("\n"," ")
            })
    if args.dry_run and len(items) > args.dry_run:
        items = items[:args.dry_run]

    print(f"Loaded {len(items)} answers from {len(files)} files.")

    pipe = pipeline("text-classification", model="Intel/polite-guard", device_map="auto", truncation=True)

    # 批处理
    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "model_in_file", "query_id", "polite_guard_score"])
        # 按批计算
        bs = max(1, args.batch_size)
        for start in tqdm(range(0, len(items), bs), desc="polite-guard"):
            batch = items[start:start+bs]
            texts = [x["answer"] for x in batch]
            outputs = pipe(texts, truncation=True)
            # outputs 与 texts 等长
            for rec, out in zip(batch, outputs):
                label = (out.get("label") or "").lower()
                s = float(out.get("score", 0.0))
                p_polite = s if label == "polite" else (1.0 - s)
                score = 2.0 * p_polite - 1.0
                w.writerow([rec["file"], rec["model_in_file"], rec["query_id"], score])

    print(f"Done. Wrote -> {args.out}")

if __name__ == "__main__":
    main()