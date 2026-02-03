from datasets import load_dataset
import json

dataset = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev", split="dev")

with open("data/iwslt26/dev.jsonl", "w") as f:
    for line in dataset:
        f.write(json.dumps({
            "audio_path": line["audio_path"],
            "src_text": line["src_text"],
            "tgt_text": line["tgt_text"],
            "tgt_system": line["tgt_system"],
            "doc_id": line["doc_id"],
            "score": line["score"],
            "src_lang": line["src_lang"],
            "tgt_lang": line["tgt_lang"],
        }) + "\n")