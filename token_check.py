from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "44David/qwen-0.5b-reasoning-v2",
    trust_remote_code=True,
)

for text in ["<think>", "</think>"]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(text, ids, tokenizer.convert_ids_to_tokens(ids))
