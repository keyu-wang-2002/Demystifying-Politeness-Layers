#!/usr/bin/env python3
"""
Politeness scoring (FINAL):
- Chunked API calls
- Strict JSON array output
- Resume (skip done chunks in existing CSV)
- Error classification + logging
- Save raw outputs per chunk
- tqdm progress bars: overall + per-chunk
- Capped retry backoff (max 2s sleep between retries)

CSV columns: file, model_in_file, query_id, chunk_id, score
"""

import argparse, os, io, glob, json, csv, re, sys, time, math, datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI
from tqdm import tqdm

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing *.json files to score")
    ap.add_argument("--out", type=str, default="scores.csv", help="Output CSV path")
    ap.add_argument("--chunk_size", type=int, default=100, help="Answers per API call")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    ap.add_argument("--rpm", type=int, default=200, help="Requests per minute (soft throttle)")
    ap.add_argument("--max_retries", type=int, default=6, help="Max retries per chunk on retriable errors")
    ap.add_argument("--temperature", type=float, default=0.0, help="Judge temperature")
    ap.add_argument("--log", type=str, default="scores.log", help="Log file path")
    ap.add_argument("--raw_dir", type=str, default="raw_outputs", help="Folder to save raw model outputs per chunk")
    ap.add_argument("--dry_run", type=int, default=0, help="Only score first K records (0 = off)")
    return ap.parse_args()

def read_records(path: str) -> List[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    # Try JSON array / dict(list)
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
    # Fallback: JSONL
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            continue
    return recs

def build_prompt(sentences: List[str]) -> str:
    N = len(sentences)
    header = f"""You are a politeness judge. Rate each response on a continuous scale from -1 (very impolite) to 1 (very polite).
Reply with ONLY a JSON array of {N} numbers (no text, no labels, no code fences).

Examples (do not echo these):

Response: "Can you look at this when you have a chance?"
Politeness score: 0.7
Response: "I would appreciate it if you could clarify this point."
Politeness score: 0.8
Response: "Please fix this error."
Politeness score: 0.0
Response: "Why didn't you follow the instructions properly?"
Politeness score: -0.5
Response: "Shut up, you clearly don't understand this."
Politeness score: -1.0

Now evaluate the following {N} responses in order:
"""
    numbered = "\n".join([
        f"{i+1}. Response: \"{s.replace('\\', '\\\\').replace('\"','\\\"')}\""
        for i, s in enumerate(sentences)
    ])
    tail = f"""\nReturn ONLY this (replace with numbers): 
[0.7, -0.5, 0.0, ...]  # length = {N}"""
    return header + numbered + tail

def parse_scores_strict(text: str, n: int) -> List[Optional[float]]:
    # Try strict JSON parsing first
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and len(arr) >= n:
            out = []
            for v in arr[:n]:
                if isinstance(v, (int, float)):
                    fv = float(v)
                    if -1.2 <= fv <= 1.2:
                        out.append(max(-1.0, min(1.0, fv)))
                    else:
                        out.append(None)
                else:
                    out.append(None)
            return out
    except Exception:
        pass
    # Fallback: regex floats
    nums = [float(x) for x in FLOAT_RE.findall(text)]
    out = []
    for v in nums[:n]:
        out.append(max(-1.0, min(1.0, v)) if -1.2 <= v <= 1.2 else None)
    if len(out) < n:
        out += [None] * (n - len(out))
    return out

@dataclass
class ClientCfg:
    model: str
    rpm: int
    max_retries: int
    temperature: float

def classify_error(e: Exception) -> str:
    msg = str(e).lower()
    if "rate limit" in msg or "too many requests" in msg or "rpm" in msg:
        if "per minute" in msg or "requests per minute" in msg:
            return "rate_limit_rpm"
        if "per day" in msg or "requests per day" in msg or "rpd" in msg:
            return "rate_limit_rpd"
        return "rate_limit"
    if "tokens per minute" in msg or "tpm" in msg:
        return "rate_limit_tpm"
    if "maximum context length" in msg or "context length" in msg or "token limit" in msg:
        return "context_length"
    if "insufficient_quota" in msg or "quota" in msg:
        return "quota_exceeded"
    if "invalid api key" in msg or "authentication" in msg or "unauthorized" in msg:
        return "auth_error"
    if "timeout" in msg or "timed out" in msg:
        return "network_timeout"
    if "connection" in msg or "network" in msg:
        return "network_error"
    return "unknown_error"

class JudgeClient:
    def __init__(self, cfg: ClientCfg, logf):
        self.cfg = cfg
        self.client = OpenAI()
        self.min_interval = 60.0 / max(1, cfg.rpm)
        self._last_call = 0.0
        self.logf = logf

    def _throttle(self):
        now = time.time()
        wait = self.min_interval - (now - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def score_chunk(self, sentences: List[str]) -> str:
        prompt = build_prompt(sentences)
        delay = 1.5
        for attempt in range(self.cfg.max_retries + 1):
            try:
                self._throttle()
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                    max_tokens=max(12 * len(sentences), 256)
                )
                self._last_call = time.time()
                return resp.choices[0].message.content.strip()
            except Exception as e:
                kind = classify_error(e)
                msg = f"[{now_ts()}] attempt {attempt} error ({kind}): {e}"
                print(msg)
                print(msg, file=self.logf); self.logf.flush()

                # If quota/day/tpm limit reached, stop immediately
                if kind in {"rate_limit_rpd", "quota_exceeded", "rate_limit_tpm"}:
                    print("⚠️ Quota exhausted or TPM limit reached. Please retry tomorrow.")
                    sys.exit(5)

                if attempt >= self.cfg.max_retries:
                    raise

                time.sleep(min(delay, 10.0))
                delay *= 1.8

def load_existing_chunks(out_path: str) -> set:
    done = set()
    if not os.path.exists(out_path):
        return done
    with io.open(out_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["chunk_id"]))
            except Exception:
                continue
    return done

def main():
    args = parse_args()
    log_mode = "a" if os.path.exists(args.log) else "w"
    os.makedirs(args.raw_dir, exist_ok=True)

    with io.open(args.log, log_mode, encoding="utf-8") as logf:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))
        all_items = []
        for fp in files:
            for r in read_records(fp):
                all_items.append({
                    "file": os.path.basename(fp),
                    "model": r.get("model", ""),
                    "query_id": r.get("query_id", ""),
                    "answer": str(r.get("answer", "")).replace("\n", " ")
                })
        n = len(all_items)
        if args.dry_run and n > args.dry_run:
            all_items = all_items[:args.dry_run]
            n = len(all_items)
        if not n:
            print("No records found.")
            sys.exit(3)

        chunk_size = max(1, args.chunk_size)
        chunk_count = math.ceil(n / chunk_size)

        # Print summary BEFORE any API call
        print(f"Read {n} records, split into {chunk_count} chunks, chunk_size={chunk_size}")

        done_chunks = load_existing_chunks(args.out)

        cfg = ClientCfg(args.model, args.rpm, args.max_retries, args.temperature)
        judge = JudgeClient(cfg, logf)

        write_header = not os.path.exists(args.out)
        mode = "a" if not write_header else "w"

        with io.open(args.out, mode, encoding="utf-8", newline="") as f_out:
            w = csv.writer(f_out)
            if write_header:
                w.writerow(["file", "model_in_file", "query_id", "chunk_id", "score"])

            # overall progress bar
            already_done = 0
            for cid in done_chunks:
                start = cid * chunk_size
                end = min((cid + 1) * chunk_size, n)
                already_done += max(0, end - start)

            with tqdm(total=n, desc="overall", unit="resp", initial=already_done) as pbar_overall:
                for chunk_id in range(chunk_count):
                    if chunk_id in done_chunks:
                        continue

                    start = chunk_id * chunk_size
                    chunk = all_items[start:start + chunk_size]
                    sentences = [x["answer"] for x in chunk]

                    print(f"[run ] chunk {chunk_id} · {len(sentences)} responses …")
                    raw = judge.score_chunk(sentences)

                    # Save raw output for this chunk
                    raw_path = os.path.join(args.raw_dir, f"chunk_{chunk_id:04d}.txt")
                    with io.open(raw_path, "w", encoding="utf-8") as rf:
                        rf.write(raw)

                    scores = parse_scores_strict(raw, len(sentences))

                    # per-chunk progress bar
                    for rec, sc in tqdm(list(zip(chunk, scores)), desc=f"chunk {chunk_id}", unit="resp", leave=False):
                        w.writerow([rec["file"], rec["model"], rec["query_id"], chunk_id, sc])
                        pbar_overall.update(1)

                    f_out.flush()
                    print(f"[done] chunk {chunk_id} finished, raw -> {raw_path}")

        print(f"Completed: {n} records, {chunk_count} chunks. CSV -> {args.out}")

if __name__ == "__main__":
    main()
