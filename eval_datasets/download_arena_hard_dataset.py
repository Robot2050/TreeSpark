import json
from pathlib import Path
from huggingface_hub import hf_hub_download

src = Path(hf_hub_download(
    repo_id="lmarena-ai/arena-hard-auto",
    filename="data/arena-hard-v2.0/question.jsonl",
    repo_type="dataset",
))
out = Path("eval_datasets/arena-hard-v2.jsonl")

with src.open("r", encoding="utf-8") as f, out.open("w", encoding="utf-8") as g:
    for line in f:
        row = json.loads(line)
        g.write(json.dumps({"turns": [row["prompt"]]}, ensure_ascii=False) + "\n")
