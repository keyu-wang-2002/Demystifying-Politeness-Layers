from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import copy

login("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")

model_name = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

def prune_selected_layers(model, prune_list):
    num_layers = len(model.model.layers)
    layers_to_remove = prune_list
    remove_n_layers = len(layers_to_remove)

    if remove_n_layers <= 0:
        return model
    if remove_n_layers >= num_layers:
        raise ValueError(f"Cannot remove all layers. Model has {num_layers} layers, attempted to remove {remove_n_layers}")

    layers_to_remove.sort(reverse=True)
    print("remove layers: " + str(layers_to_remove))

    for index in layers_to_remove:
        del model.model.layers[index]

    model.config.num_hidden_layers = len(model.model.layers)
    return model


input_file = "./data/text_random_50.json"

with open(input_file, "r", encoding="utf-8") as f:
    prompt_lines = [json.loads(line) for line in f]


baseline_model = copy.deepcopy(base_model)
baseline_output_file = "./results/LLaMA3.1-8B_Baseline.json"
baseline_results = []

for idx, data in enumerate(prompt_lines):
    query = data["text"]
    inputs = tokenizer(query, return_tensors="pt").to(baseline_model.device)

    with torch.no_grad():
        outputs = baseline_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    baseline_results.append({
        "model": "LLaMA3.1-8B_Baseline",
        "query_id": idx,
        "query": query,
        "answer": answer
    })

with open(baseline_output_file, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, ensure_ascii=False, indent=4)

print(f"Baseline results saved to {baseline_output_file}")

for prune_layer in range(0, 31):   
    model = copy.deepcopy(base_model)
    compressed_model = prune_selected_layers(model, [prune_layer])

    output_file = f"./results/LLaMA3.1-8B_Prune-{prune_layer}th-Layer.json"

    results = []

    for idx, data in enumerate(prompt_lines):
        query = data["text"]
        inputs = tokenizer(query, return_tensors="pt").to(compressed_model.device)

        with torch.no_grad():
            outputs = compressed_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                use_cache=False
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "model": f"LLaMA3.1-8B_Prune-{prune_layer}th-Layer",
            "query_id": idx,
            "query": query,
            "answer": answer
        })

    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results saved to {output_file}")