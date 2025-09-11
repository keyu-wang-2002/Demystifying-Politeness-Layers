# Generated from Untitled.ipynb — All comments are in English.
pip install openai


# ---- cell separator ----

from openai import OpenAI

MODEL = "gpt-4o-mini"
# 初始化 client（填你的 API Key）
client = OpenAI(api_key="", max_retries=0)

# ---- cell separator ----

import os, time, json, random, re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from openai import OpenAI
from openai._exceptions import RateLimitError, APIStatusError
# ---- Simple JSON cache （避免重复请求，抗429）----
CACHE_PATH = "judge_cache.jsonl"
_cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            _cache[(rec["model"], rec["prompt_hash"])] = rec["output"]

def save_cache(model: str, prompt_hash: str, output: str):
    if (model, prompt_hash) in _cache:
        return
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"model": model, "prompt_hash": prompt_hash, "output": output}, ensure_ascii=False)+"\n")
    _cache[(model, prompt_hash)] = output

def prompt_hash_fn(text: str) -> str:
    # 简单 hash（可换成 hashlib.md5）
    return str(abs(hash(text)) % (10**12))

# ---- Backoff ----
def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 20.0):
    wait = min(cap, base * (2 ** attempt)) + random.uniform(0, 0.5 * (attempt + 1))
    time.sleep(wait)

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def parse_score(text: str, lo=-1.0, hi=1.0) -> Optional[float]:
    if not text:
        return None
    m = _num_re.search(text)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return max(lo, min(hi, v))
    except:
        return None

def call_openai_with_retry(prompt: str, max_tries=6, timeout=2000.0) -> str:
    for attempt in range(max_tries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=6,
                timeout=timeout
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            if attempt == max_tries-1: raise
            _backoff_sleep(attempt)
        except APIStatusError as e:
            sc = getattr(e, "status_code", None)
            if sc and 500 <= sc < 600 and attempt < max_tries-1:
                _backoff_sleep(attempt); continue
            raise

# ---- cell separator ----

import math, re, time

def build_multi_prompt(sent_list, cfg: PromptConfig) -> str:
    """把多条句子合并成一次评分请求，返回 prompt。"""
    instr = cfg.instruction.format(low=cfg.scale_low, high=cfg.scale_high)
    fmt   = cfg.format_rule.format(low=cfg.scale_low, high=cfg.scale_high)
    examples = cfg.few_shots or []
    if cfg.shuffle_examples:
        examples = examples.copy(); random.shuffle(examples)

    ex_txt = "\n".join([
        f'Response: "{e["text"]}"\nPoliteness score: {e["score"]}\n'
        for e in examples
    ])

    numbered = "\n".join([f'{i+1}. "{s}"' for i, s in enumerate(sent_list)])
    # 让模型严格逐行输出“index: score”
    return f"""{instr}

{fmt}

Examples:
{ex_txt}

Now evaluate the following new responses.
For each i-th response, output exactly one line in the format:
i: <numeric_score_in_[{cfg.scale_low},{cfg.scale_high}]>

Responses:
{numbered}

Politeness scores:
""".strip()


_pair_re = re.compile(r"^\s*(\d+)\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")

def parse_multi_scores(text: str, n: int, lo=-1.0, hi=1.0):
    """
    解析 “i: score” 多行输出，返回长度为 n 的分数列表（缺失记 None），并裁剪到 [lo, hi]。
    """
    out = [None] * n
    if not text:
        return out
    for line in text.strip().splitlines():
        m = _pair_re.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n:
            try:
                v = float(m.group(2))
                v = max(lo, min(hi, v))
                out[idx-1] = v
            except:
                pass
    return out


from tqdm import tqdm

def evaluate_df_batched(df: pd.DataFrame, cfg: PromptConfig, text_col="text",
                        batch_size=8, rpm=3, verbose=True) -> pd.DataFrame:
    texts = df[text_col].astype(str).tolist()
    preds = [None] * len(texts)

    interval = 60.0 / max(1, rpm)
    num_batches = math.ceil(len(texts) / batch_size)

    for b in tqdm(range(num_batches), desc="Batches", leave=False):
        start = b * batch_size
        end   = min(len(texts), (b+1) * batch_size)
        chunk = texts[start:end]

        prompt = build_multi_prompt(chunk, cfg)
        h = prompt_hash_fn(MODEL + "||" + prompt)
        if (MODEL, h) in _cache:
            text = _cache[(MODEL, h)]
        else:
            text = call_openai_with_retry(prompt)
            save_cache(MODEL, h, text)

        batch_scores = parse_multi_scores(text, n=len(chunk),
                                          lo=cfg.scale_low, hi=cfg.scale_high)
        preds[start:end] = batch_scores

        if verbose:
            got = sum(s is not None for s in batch_scores)
            tqdm.write(f"  Batch {b+1}/{num_batches} -> {got}/{len(chunk)} scores")

        if b < num_batches - 1:
            time.sleep(interval)
    out = df.copy()
    out["pred"] = preds
    return out



def regression_metrics(df: pd.DataFrame, target_col="target_score", pred_col="pred"):
    d = df.dropna(subset=[target_col, pred_col])
    if len(d) == 0:
        return {"spearman": None, "pearson": None, "mae": None}
    from scipy.stats import spearmanr, pearsonr
    return {
        "spearman": float(spearmanr(d[target_col], d[pred_col]).correlation),
        "pearson":  float(pearsonr(d[target_col], d[pred_col]).statistic),
        "mae":      float(np.mean(np.abs(d[target_col] - d[pred_col])))
    }


# ---- cell separator ----

@dataclass
class PromptConfig:
    scale_low: float = -1.0
    scale_high: float = 1.0
    instruction: str = (
        "You are a politeness judge. Evaluate the politeness of the response "
        "on a continuous scale from {low} (very impolite) to {high} (very polite). "
        "Respond with only the numeric score, no explanation. If uncertain, output your best estimate."
    )
    format_rule: str = "Output only a number in [{low}, {high}]. Nothing else."
    few_shots: List[Dict[str, Any]] = None      # [{'text':..., 'score':...}, ...]
    shuffle_examples: bool = False              # 控制 few-shot 顺序打乱

def build_prompt(sentence: str, cfg: PromptConfig) -> str:
    instr = cfg.instruction.format(low=cfg.scale_low, high=cfg.scale_high)
    fmt   = cfg.format_rule.format(low=cfg.scale_low, high=cfg.scale_high)
    examples = cfg.few_shots or []
    if cfg.shuffle_examples:
        examples = examples.copy(); random.shuffle(examples)

    ex_txt = "\n".join([
        f'Response: "{e["text"]}"\nPoliteness score: {e["score"]}\n'
        for e in examples
    ])
    prompt = f"""{instr}

{fmt}

Examples:
{ex_txt}
Now evaluate the following new response:
Response: "{sentence}"
Politeness score:
"""
    return prompt.strip()

# ---- cell separator ----

toy = pd.DataFrame({
    "text": [
        "Give me the answer right now!",
        "Could you kindly help me with this problem?",
        "This problem is 2+2, it equals 4.",
        "What a dumb question, it's obviously 8.",
        "Please review this when you have time, thank you!"
    ],
    # 人工目标分（用于 sanity 对比；不影响真实学术评测）
    "target_score": [-0.7, 0.7, 0.0, -1.0, 0.8]
})
toy


# ---- cell separator ----

!pip install convokit

# ---- cell separator ----

from convokit import Corpus, download

# Wikipedia politeness
wiki = Corpus(filename=download("wikipedia-politeness-corpus"))

# StackExchange politeness
se = Corpus(filename=download("stack-exchange-politeness-corpus"))

#说实话这一大堆我的注意力机制最后选中了wiki
# 查看一条
utt = list(wiki.iter_utterances())[0]
print(utt.text, utt.meta)


# ---- cell separator ----

# pip install convokit pandas numpy
import os, json, random
import pandas as pd
import numpy as np

# ========== 参数 ==========
SAMPLE_SIZE = 40         # 想要导出的样本数（小样本便于你先跑完后续测试）
BINS = 5                 # 分 5 桶做分层均匀采样
OUT_CSV = "stanford_politeness_sample.csv"
USE_CONVOKIT_DOWNLOAD = True  # True: 直接在线用 convokit 下载；False: 用本地路径

# 如果走本地路径，把这两个指向你解压后的目录（里面有 utterances.jsonl）
LOCAL_WIKI_DIR = "./wikipedia-politeness-corpus"
LOCAL_SE_DIR   = "./stack-exchange-politeness-corpus"


def load_corpus_utterances_from_convokit(name: str):
    """用 convokit 在线下载并加载 utterances.jsonl，返回 DataFrame(text, raw_score, source)"""
    from convokit import Corpus, download
    c = Corpus(filename=download(name))
    rows = []
    for utt in c.iter_utterances():
        meta = utt.meta or {}
        text = (utt.text or "").strip()
        if not text:
            continue
        # 这些键名在不同版本里可能有差异，做个鲁棒读取
        # 候选：'Normalized Score', 'Average Score', 'Score', 'Binary'
        score = None
        for key in ["Normalized Score", "Average Score", "Score"]:
            if key in meta:
                try:
                    score = float(meta[key])
                    break
                except Exception:
                    pass
        if score is None and "Binary" in meta:
            # Binary 通常是 {-1, 1}；如果是 {0,1} 也做映射
            b = meta["Binary"]
            try:
                b = int(b)
            except Exception:
                try:
                    b = int(float(b))
                except Exception:
                    b = None
            if b is not None:
                if b in (-1, 1):
                    score = float(b)
                elif b in (0, 1):
                    score = -1.0 if b == 0 else 1.0

        if score is None:
            continue  # 没有可用分数就跳过

        rows.append({"text": text, "raw_score": float(score), "source": name})
    return pd.DataFrame(rows)


def load_corpus_utterances_from_local(dir_path: str, source_name: str):
    """从本地 utterances.jsonl 读取，返回 DataFrame(text, raw_score, source)"""
    ujson = os.path.join(dir_path, "utterances.jsonl")
    if not os.path.isfile(ujson):
        raise FileNotFoundError(f"Not found: {ujson}")
    rows = []
    with open(ujson, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            meta = obj.get("meta") or {}
            score = None
            for key in ["Normalized Score", "Average Score", "Score"]:
                if key in meta:
                    try:
                        score = float(meta[key])
                        break
                    except Exception:
                        pass
            if score is None and "Binary" in meta:
                b = meta["Binary"]
                try:
                    b = int(b)
                except Exception:
                    try:
                        b = int(float(b))
                    except Exception:
                        b = None
                if b is not None:
                    if b in (-1, 1):
                        score = float(b)
                    elif b in (0, 1):
                        score = -1.0 if b == 0 else 1.0
            if score is None:
                continue
            rows.append({"text": text, "raw_score": float(score), "source": source_name})
    return pd.DataFrame(rows)


# ========== 1) 加载 Wiki + SE ==========
if USE_CONVOKIT_DOWNLOAD:
    wiki_df = load_corpus_utterances_from_convokit("wikipedia-politeness-corpus")
    se_df   = load_corpus_utterances_from_convokit("stack-exchange-politeness-corpus")
else:
    wiki_df = load_corpus_utterances_from_local(LOCAL_WIKI_DIR, "wikipedia-politeness-corpus")
    se_df   = load_corpus_utterances_from_local(LOCAL_SE_DIR, "stack-exchange-politeness-corpus")

raw = pd.concat([wiki_df, se_df], ignore_index=True)
raw = raw.dropna(subset=["raw_score"]).drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"Loaded rows: {len(raw)}")
print(raw.head(3))

# ========== 2) 归一化到 [-1, 1] ==========
# 说明：Convokit 的 'Normalized Score' 可能越界（比如你看到的 -1.12…），
# 所以我们对全体 raw_score 做一次 min-max → [-1,1]，保证区间一致。
xmin, xmax = float(raw["raw_score"].min()), float(raw["raw_score"].max())
def minmax_to_minus1_1(x, xmin, xmax):
    if xmax == xmin: return 0.0
    return 2.0 * ((x - xmin) / (xmax - xmin)) - 1.0

raw["score"] = raw["raw_score"].apply(lambda v: minmax_to_minus1_1(v, xmin, xmax))
raw = raw[["text", "score", "source"]]

# ========== 3) 分层均匀采样（按 score 分桶） ==========
# 目标：每个区间都抽一些，避免全是中等分的句子
raw["bin"] = pd.qcut(raw["score"], q=BINS, labels=False, duplicates="drop")
per_bin = max(1, SAMPLE_SIZE // max(1, raw["bin"].nunique()))
samples = []
for b in sorted(raw["bin"].dropna().unique()):
    chunk = raw[raw["bin"] == b]
    k = min(per_bin, len(chunk))
    samples.append(chunk.sample(n=k, random_state=42))
sampled = pd.concat(samples, ignore_index=True)

# 如果总数不足 SAMPLE_SIZE，再随机补齐
if len(sampled) < SAMPLE_SIZE and len(raw) > len(sampled):
    remain = raw.drop(sampled.index)
    need = min(SAMPLE_SIZE - len(sampled), len(remain))
    if need > 0:
        sampled = pd.concat([sampled, remain.sample(n=need, random_state=123)], ignore_index=True)

# ========== 4) 导出为你需要的格式（text, score） ==========
final = sampled[["text", "score"]].reset_index(drop=True)
final.to_csv(OUT_CSV, index=False, encoding="utf-8")

print(f"\nSaved: {OUT_CSV}  (rows={len(final)})")
display(final.head(10))

# 快速看一下分布
print("\nScore summary:\n", final["score"].describe())
print("\nApprox bins coverage:")
print(final.assign(bin=pd.cut(final["score"], bins=BINS, labels=False)).groupby("bin").size())


# ---- cell separator ----

def judge_one(sentence: str, cfg: PromptConfig) -> Optional[float]:
    prompt = build_prompt(sentence, cfg)
    h = prompt_hash_fn(MODEL + "||" + prompt)
    if (MODEL, h) in _cache:
        text = _cache[(MODEL, h)]
    else:
        text = call_openai_with_retry(prompt)
        save_cache(MODEL, h, text)
    return parse_score(text, lo=cfg.scale_low, hi=cfg.scale_high)

def evaluate_df(df: pd.DataFrame, cfg: PromptConfig, text_col="text", target_col="target_score") -> pd.DataFrame:
    preds = []
    for s in tqdm(df[text_col].tolist()):
        preds.append(judge_one(s, cfg))
    out = df.copy()
    out["pred"] = preds
    return out

def regression_metrics(df: pd.DataFrame, target_col="target_score", pred_col="pred"):
    # 适用于连续分
    d = df.dropna(subset=[target_col, pred_col])
    spearman = spearmanr(d[target_col], d[pred_col]).correlation
    pearson_ = pearsonr(d[target_col], d[pred_col]).statistic
    mae = float(np.mean(np.abs(d[target_col]-d[pred_col])))
    return {"spearman": spearman, "pearson": pearson_, "mae": mae}

def classification_metrics(df: pd.DataFrame, target_col="label", pred_col="pred", thresh=0.0):
    # 若你用二分类（polite=1, impolite=0 或 -1/1），就用这个
    d = df.dropna(subset=[target_col, pred_col]).copy()
    d["pred_label"] = (d[pred_col] >= thresh).astype(int)
    acc = (d["pred_label"]==d[target_col]).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(d[target_col], d["pred_label"])
    return {"acc": float(acc), "f1": float(f1)}


# ---- cell separator ----

fewshot_bank_A = [
    {"text": "Can you look at this when you have a chance?", "score": 0.7},
    {"text": "I would appreciate it if you could clarify this point.", "score": 0.8},
    {"text": "Please fix this error.", "score": 0.0},
    {"text": "Why didn't you follow the instructions properly?", "score": -0.5},
    {"text": "Shut up, you clearly don't understand this.", "score": -1.0},
]


# ---- cell separator ----

cfg = PromptConfig(
    scale_low=-1.0, scale_high=1.0,
    few_shots=fewshot_bank_A,
    shuffle_examples=False
)

toy_out = evaluate_df(toy, cfg)
print(toy_out)
print(regression_metrics(toy_out))

# ---- cell separator ----

instruction_set = [
    # 更严格格式约束
    ("strict", "You are a politeness judge. Rate on {low}..{high}. Reply with ONLY a number. If you output anything else, you fail. Return a number."),
    # 稍松
    ("mild",   "You are a politeness judge. Rate on {low}..{high}. Output one numeric value only.")
]

format_set = [
    ("hard", "Output only a number in [{low}, {high}]. No words, no explanation, no units."),
    ("soft", "Output a numeric score in [{low}, {high}].")
]

shots_variants = [
    ("A5", fewshot_bank_A[:5]),
    ("A3", fewshot_bank_A[:3]),
]
import time#我的api限制哈
def sweep(df: pd.DataFrame, batch_size=8, rpm=3):
    rows = []
    combos = [(ins_name, ins, fmt_name, fmt, shots_name, shots)
              for ins_name, ins in instruction_set
              for fmt_name, fmt in format_set
              for shots_name, shots in shots_variants]

    for (ins_name, ins, fmt_name, fmt, shots_name, shots) in tqdm(combos, desc="Prompt configs"):
        cfg = PromptConfig(
            scale_low=-1.0, scale_high=1.0,
            instruction=ins,
            format_rule=fmt,
            few_shots=shots,
            shuffle_examples=True
        )
        tqdm.write(f"Running ins={ins_name}, fmt={fmt_name}, shots={shots_name} ...")
        out = evaluate_df_batched(df, cfg, batch_size=batch_size, rpm=rpm, verbose=False)
        metrics = regression_metrics(out, target_col="target_score", pred_col="pred")
        rows.append({
            "ins": ins_name,
            "fmt": fmt_name,
            "shots": shots_name,
            **metrics
        })
    return pd.DataFrame(rows).sort_values(by=["spearman","pearson"], ascending=False)

leaderboard = sweep(toy, batch_size=4, rpm=3)
leaderboard

# ---- cell separator ----

def inspect_errors(df_out: pd.DataFrame, topk=10):
    df2 = df_out.dropna(subset=["target_score","pred"]).copy()
    df2["abs_err"] = (df2["target_score"] - df2["pred"]).abs()
    return df2.sort_values("abs_err", ascending=False).head(topk)[["text","target_score","pred","abs_err"]]

inspect_errors(toy_out, topk=5)

# ---- cell separator ----

df_real = final.rename(columns={"score": "target_score"})
df_real.head()

# ---- cell separator ----

leaderboard = sweep(df_real, batch_size=20, rpm=3)
leaderboard
