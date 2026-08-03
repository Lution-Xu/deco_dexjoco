import argparse
import os

import torch
import yaml

from dexjoco_constants_unify import UNIFIED_TASKS


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path, base_dir=None):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    if base_dir is not None:
        candidate = os.path.abspath(os.path.join(base_dir, path))
        if os.path.exists(candidate):
            return candidate
    cwd_candidate = os.path.abspath(path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(REPO_ROOT, path))


def load_prompts(prompt_config_path):
    with open(prompt_config_path, "r") as f:
        prompt_cfg = yaml.safe_load(f)
    task_order = prompt_cfg.get("task_order")
    prompts = prompt_cfg.get("prompts", {})
    if task_order != list(UNIFIED_TASKS):
        raise ValueError(
            "Prompt task_order must exactly match dexjoco_constants_unify.UNIFIED_TASKS. "
            f"Expected {list(UNIFIED_TASKS)}, got {task_order}"
        )
    missing = [task for task in task_order if task not in prompts]
    if missing:
        raise ValueError(f"Prompt config is missing prompts for: {missing}")
    return task_order, [prompts[task] for task in task_order]


def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask[:, :, None].to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def main():
    parser = argparse.ArgumentParser(description="Precompute offline T5 features for DECO_Text DexJoCo unify.")
    parser.add_argument("--config", default="config2/deco_text_dexjoco_unify.yaml")
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-name", default="google-t5/t5-base")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache/model cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load tokenizer/model files that already exist locally.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prompt_config = args.prompt_config or config["model"].get(
        "prompt_config_path",
        "config2/deco_text_dexjoco_prompts.yaml",
    )
    prompt_config_path = resolve_path(prompt_config, base_dir=config_dir)
    output_path = args.output or config["model"]["text_feature_path"]
    output_path = resolve_path(output_path, base_dir=config_dir)

    task_names, prompts = load_prompts(prompt_config_path)

    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise ImportError(
            "Install transformers plus the T5 tokenizer dependencies first. "
            "In this environment that usually means: pip install sentencepiece protobuf"
        ) from exc

    try:
        local_files_only = args.local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1"
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            local_files_only=local_files_only,
        )
        model = T5EncoderModel.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            local_files_only=local_files_only,
        ).to(args.device)
    except Exception as exc:
        message = str(exc).lower()
        if "sentencepiece" in message or "protobuf" in message:
            raise RuntimeError(
                "T5 tokenizer/model loading needs extra dependencies. "
                "Install them with: pip install -r requirements_deco_text.txt"
            ) from exc
        raise
    model.eval()

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, return_dict=True)
    features = masked_mean_pool(outputs.last_hidden_state, encoded["attention_mask"]).cpu()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "features": features,
            "task_names": task_names,
            "prompts": dict(zip(task_names, prompts)),
            "model_name": args.model_name,
            "pooling": "masked_mean",
            "max_length": args.max_length,
        },
        output_path,
    )
    print(f"Wrote {tuple(features.shape)} T5 features to {output_path}")


if __name__ == "__main__":
    main()
