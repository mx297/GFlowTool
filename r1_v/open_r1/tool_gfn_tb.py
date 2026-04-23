import os
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Optional

from datasets import load_dataset
from PIL import Image
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from r1_v.open_r1.trainer.tool_vllm_gfn_tb_trainer_safe import Qwen2VLGFNTBVLLMTrainer
from r1_v.open_r1.utils.tokenizer_guard import install_explicit_truncation_guard


@dataclass
class GFNTBScriptArguments(ScriptArguments):
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    finetune_mode: str = field(default="lora")
    freeze_vision_tower: bool = field(default=True)
    max_rounds: int = field(default=4)
    reward_accuracy_weight: float = field(default=1.0)
    reward_format_weight: float = field(default=0.25)
    reward_epsilon: float = field(default=1e-4)
    replay_buffer_size: int = field(default=2048)
    replay_sampling: str = field(default="prioritized")
    replay_priority_alpha: float = field(default=1.0)
    rollout_sync_interval: int = field(default=8)
    log_reward_clip_min: float = field(default=-9.21)
    logZ_init: float = field(default=0.0)
    logZ_lr: float = field(default=1e-2)
    logz_hidden_dim: int = field(default=512)
    logz_dropout: float = field(default=0.0)
    logz_max_length: int = field(default=512)
    logz_pooling: str = field(default="mean")
    logz_detach_base_embeddings: bool = field(default=True)
    query_key: Optional[str] = field(default="question")
    controller_addr: Optional[str] = field(
        default=None,
        metadata={"help": "Address of the tool controller. Defaults to the locally recorded controller address."},
    )
    top_p: float = field(default=1.0)
    top_k: int = field(default=-1)
    trajectory_log_path: Optional[str] = field(default=None)
    trajectory_log_every_steps: int = field(default=50)
    trajectory_log_num_samples: int = field(default=4)
    trajectory_log_include_turn_records: bool = field(default=False)
    trajectory_log_max_text_chars: int = field(default=2000)


STRICT_SYSTEM_PROMPT = """You are a visual assistant capable of generating and solving steps for chart-based reasoning. Your goal is to answer chart-related questions. You can rely on your own capabilities or use external tools to assist in solving. Here are the available actions:
- **OCR**: Extracts text from an image. Example: `{"name": "OCR", "arguments": {"image": "img_1"}}`
- **Point**: Identifies a point in the image based on description and returns coordinates. Example: `{"name": "Point", "arguments": {"image": "img_1", "param": "x-axis value 1970"}}`
- **ZoomInSubfigure**: Crops the image to the specified subfigure. Example: `{"name": "ZoomInSubfigure", "arguments": {"image": "img_1", "param": "Downstream vs. Concept: Toy"}}`
- **SegmentRegionAroundPoint**: Segments a region around a given point. Example: `{"name": "SegmentRegionAroundPoint", "arguments": {"image": "img_1", "param": "x=\\"21.5\\" y=\\"28.5\\""}}`
- **DrawHorizontalLineByY**: Draws a horizontal line at a given y-coordinate. Example: `{"name": "DrawHorizontalLineByY", "arguments": {"image": "img_1", "param": "y=28.5"}}`
- **DrawVerticalLineByX**: Draws a vertical line at a given x-coordinate. Example: `{"name": "DrawVerticalLineByX", "arguments": {"image": "img_1", "param": "x=21.5"}}`
- **Terminate**: Ends the task and provides the final answer. Example: `{"name": "Terminate", "arguments": {"ans": "1985"}}`

To solve the problem:
1. Select actions from the provided tools list, combining them logically and building on previous steps. Call one action at a time, using its output for the next.
2. To use `SegmentRegionAroundPoint`, `DrawHorizontalLineByY`, or `DrawVerticalLineByX`, first call \"Point\" to get coordinates for further actions.

Your output should be in a JSON format as follows:
{"thought": "the reasoning process", "actions": [{"name": "action", "arguments": {"argument1": "value1", "argument2": "value2"}}]}
"""


def _to_plain_data(value):
    if is_dataclass(value):
        return {k: _to_plain_data(v) for k, v in asdict(value).items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return {k: _to_plain_data(v) for k, v in value.to_dict().items()}
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_yaml_scalar(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "":
        return '""'
    if any(ch in text for ch in [":", "#", "\n", "\t", "{", "}", "[", "]"]) or text.strip() != text:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _dump_yaml_lines(value, indent=0):
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_yaml_scalar(item)}")
        return lines or [f"{prefix}{{}}"]
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_dump_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_yaml_scalar(item)}")
        return lines or [f"{prefix}[]"]
    return [f"{prefix}{_format_yaml_scalar(value)}"]


def save_run_args_yaml(output_dir, script_args, training_args, model_args):
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "script_args": _to_plain_data(script_args),
        "training_args": _to_plain_data(training_args),
        "model_args": _to_plain_data(model_args),
    }
    yaml_path = os.path.join(output_dir, "training_args.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_dump_yaml_lines(payload)) + "\n")


def main(script_args, training_args, model_args):
    install_explicit_truncation_guard()

    if not training_args.use_vllm:
        raise ValueError("This trainer requires --use_vllm true.")
    if script_args.finetune_mode not in {"lora", "full"}:
        raise ValueError("--finetune_mode must be one of: lora, full")

    set_seed(training_args.seed)
    save_run_args_yaml(training_args.output_dir, script_args, training_args, model_args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    dataset = load_dataset("json", data_files=script_args.dataset_name)

    def load_image_from_path(example):
        if "solution" not in example and "label" in example:
            example["solution"] = example["label"]
        if "label" in example:
            example.pop("label", None)
        image = Image.open(example["image_path"]).convert("RGBA")
        example["image"] = image
        return example

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": example[script_args.query_key]},
            ]
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": STRICT_SYSTEM_PROMPT},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example[script_args.query_key]},
                    ],
                },
            ]
        }

    if "image_path" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(load_image_from_path)
        dataset = dataset.map(make_conversation_image)
    else:
        dataset = dataset.map(make_conversation)
        if "query" in dataset[script_args.dataset_train_split].column_names:
            dataset = dataset.remove_columns("query")

    peft_config = get_peft_config(model_args) if script_args.finetune_mode == "lora" else None

    trainer = Qwen2VLGFNTBVLLMTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        controller_addr=script_args.controller_addr,
        max_rounds=script_args.max_rounds,
        reward_accuracy_weight=script_args.reward_accuracy_weight,
        reward_format_weight=script_args.reward_format_weight,
        reward_epsilon=script_args.reward_epsilon,
        replay_buffer_size=script_args.replay_buffer_size,
        replay_sampling=script_args.replay_sampling,
        replay_priority_alpha=script_args.replay_priority_alpha,
        rollout_sync_interval=script_args.rollout_sync_interval,
        freeze_vision_tower=script_args.freeze_vision_tower,
        log_reward_clip_min=script_args.log_reward_clip_min,
        logZ_init=script_args.logZ_init,
        logZ_lr=script_args.logZ_lr,
        logz_hidden_dim=script_args.logz_hidden_dim,
        logz_dropout=script_args.logz_dropout,
        logz_max_length=script_args.logz_max_length,
        logz_pooling=script_args.logz_pooling,
        logz_detach_base_embeddings=script_args.logz_detach_base_embeddings,
        top_p=script_args.top_p,
        top_k=script_args.top_k,
        trajectory_log_path=script_args.trajectory_log_path,
        trajectory_log_every_steps=script_args.trajectory_log_every_steps,
        trajectory_log_num_samples=script_args.trajectory_log_num_samples,
        trajectory_log_include_turn_records=script_args.trajectory_log_include_turn_records,
        trajectory_log_max_text_chars=script_args.trajectory_log_max_text_chars
        
    )

    print("using trainer:", trainer.__class__.__name__)
    print("finetune_mode:", script_args.finetune_mode)
    print("freeze_vision_tower:", script_args.freeze_vision_tower)
    print("max_rounds:", script_args.max_rounds)
    print("reward_accuracy_weight:", script_args.reward_accuracy_weight)
    print("reward_format_weight:", script_args.reward_format_weight)
    print("rollout_sync_interval:", script_args.rollout_sync_interval)

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GFNTBScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
