from function_helpers import *

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path



def sequence_log_probability(logits: torch.Tensor, token_ids: list[int],) -> float:
    if not token_ids:
        return float("-inf")

    log_probs = torch.log_softmax(logits[0], dim=-1)
    return float(log_probs[token_ids[0]].item())


def main():
    # variables
    rows = load_jsonl(Path("math-dataset.jsonl"))
    model = "44David/qwen-0.5b-reasoning-v2"
    device = "cpu"
    output_path = Path("routing_results.jsonl")
    activations_path = Path("routing_activations.npz")
    max_new_tokens=256

    tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    think_ids = get_token_sequence(tokenizer, "<think>")
    close_think_ids = get_token_sequence(tokenizer, "</think>")

    print("Tokenization:")
    print(f"  <think>: {think_ids}")
    print(f"  </think>: {close_think_ids}")

    if not think_ids:
        raise RuntimeError("Tokenizer produced no tokens for <think>")

    activation_store: dict[str, np.ndarray] = {}

    with output_path.open("w", encoding="utf-8") as output_file:
        for row in tqdm(rows):
            prompt = row["prompt"]

            messages = [{"role": "user", "content": prompt}]

            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)

            prompt_length = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(
                next_token_logits.float(),
                dim=-1,
            )

            first_think_token_id = think_ids[0]
            think_first_token_probability = float(
                next_token_probs[0, first_think_token_id].item()
            )

            top_values, top_indices = torch.topk(
                next_token_probs[0],
                k=10,
            )

            top_tokens = [
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "probability": float(probability),
                }
                for probability, token_id in zip(top_values, top_indices)
            ]

            predicted_token_id = int(torch.argmax(next_token_logits, dim=-1))
            predicted_token = tokenizer.decode([predicted_token_id])

            # Shape after stacking: [num_layers_plus_embedding, hidden_size]
            final_position_states = torch.stack(
                [
                    hidden_state[0, -1, :].float().cpu()
                    for hidden_state in outputs.hidden_states
                ]
            ).numpy()

            activation_store[row["id"]] = final_position_states

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            response_ids = generated_ids[0, prompt_length:]
            generated_text = tokenizer.decode(
                response_ids,
                skip_special_tokens=False,
            )

            result = {
                **row,
                "formatted_prompt": formatted_prompt,
                "generated_text": generated_text,
                "used_think": "<think>" in generated_text,
                "closed_think": "</think>" in generated_text,
                "first_generated_token_id": predicted_token_id,
                "first_generated_token": predicted_token,
                "think_first_token_id": first_think_token_id,
                "think_first_token_probability": (
                    think_first_token_probability
                ),
                "top_10_first_tokens": top_tokens,
                "prompt_token_count": prompt_length,
                "response_token_count": len(response_ids),
                "activation_key": row["id"],
            }

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    np.savez_compressed(
        activations_path,
        **activation_store,
    )

    print(f"Saved results to {output_path}")
    print(f"Saved activations to {activations_path}")


if __name__ == "__main__":
    main()
