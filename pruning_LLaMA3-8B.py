from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import copy

login("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")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, gc

model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_model():
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",        # 或 "cuda:0"
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

def prune_selected_layers(model, prune_list):
    layers = model.model.layers
    for idx in sorted(prune_list, reverse=True):
        del layers[idx]
    model.config.num_hidden_layers = len(layers)
    return model

def generate_batch(model, prompts, tag, use_cache=True):
    results = []
    for idx, data in enumerate(prompts):
        query = data["text"]
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                use_cache=use_cache
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"model": tag, "query_id": idx, "query": query, "answer": answer})
    return results

# 读数据
with open("./data/text_random_50.json", "r", encoding="utf-8") as f:
    prompt_lines = [json.loads(line) for line in f]

# 基线：只有一份模型在 GPU
model = load_model()
baseline_results = generate_batch(model, prompt_lines, "LLaMA3.1-8B_Baseline", use_cache=False)  # 与剪枝一致
with open("./results/LLaMA3.1-8B_Baseline.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=4)
del model; torch.cuda.empty_cache(); gc.collect()

# 剪一层的循环
for prune_layer in range(0, 31):
    model = load_model()               # 重新加载，而不是 deepcopy
    model.to("cpu")                    # 如需更稳，可先移到 CPU 再剪
    model = prune_selected_layers(model, [prune_layer])
    model.to("cuda")                   # 剪完再搬回 GPU

    results = generate_batch(model, prompt_lines,
                             f"LLaMA3.1-8B_Prune-{prune_layer}th-Layer",
                             use_cache=False)

    with open(f"./results/LLaMA3.1-8B_Prune-{prune_layer}th-Layer.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    del model; torch.cuda.empty_cache(); gc.collect()
