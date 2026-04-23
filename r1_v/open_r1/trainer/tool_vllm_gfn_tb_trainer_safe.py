import math
import uuid
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Optional, Union
from unittest.mock import patch
import warnings
import copy
import json
import os
from .prompt_conditioned_logz import PromptConditionedLogZ
import torch
from accelerate.utils import broadcast_object_list, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from qwen_vl_utils import process_vision_info
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer import TRAINING_ARGS_NAME
from trl.import_utils import is_vllm_available
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from .tool_generation_safe_gfn import vllm_generate_with_tool_calls_gfn
from .tool_replay_buffer import ReplayItem, ToolTrajectoryReplayBuffer
from .tool_tb_rewards import accuracy_reward_from_model_outputs, format_reward_from_model_outputs

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


class Qwen2VLGFNTBVLLMTrainer(Trainer):
    @staticmethod
    def _get_text_config_tie_word_embeddings(model_config) -> Optional[bool]:
        if model_config is None:
            return None

        text_config = getattr(model_config, "text_config", None)
        if isinstance(text_config, dict):
            return text_config.get("tie_word_embeddings")
        if text_config is not None:
            return getattr(text_config, "tie_word_embeddings", None)
        return None

    @staticmethod
    def _build_vllm_hf_overrides(model_config):
        def override_fn(vllm_config):
            text_config = getattr(vllm_config, "text_config", None)
            if isinstance(text_config, dict):
                vllm_config.text_config = PretrainedConfig(**text_config)
                text_config = vllm_config.text_config

            top_level_tie_word_embeddings = getattr(vllm_config, "tie_word_embeddings", None)
            text_config_tie_word_embeddings = (
                text_config.get("tie_word_embeddings")
                if isinstance(text_config, dict)
                else getattr(text_config, "tie_word_embeddings", None)
            )
            if not top_level_tie_word_embeddings and text_config_tie_word_embeddings:
                vllm_config.tie_word_embeddings = True
            return vllm_config

        text_config = getattr(model_config, "text_config", None)
        top_level_tie_word_embeddings = getattr(model_config, "tie_word_embeddings", None)
        text_config_tie_word_embeddings = Qwen2VLGFNTBVLLMTrainer._get_text_config_tie_word_embeddings(model_config)

        needs_text_config_normalization = isinstance(text_config, dict)
        needs_tie_word_embeddings_override = (
            not top_level_tie_word_embeddings and text_config_tie_word_embeddings
        )
        if needs_text_config_normalization or needs_tie_word_embeddings_override:
            return override_fn
        return None

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        controller_addr: Optional[str] = None,
        max_rounds: int = 4,
        reward_accuracy_weight: float = 1.0,
        reward_format_weight: float = 0.25,
        reward_epsilon: float = 1e-4,
        replay_buffer_size: int = 2048,
        replay_sampling: str = "prioritized",
        replay_priority_alpha: float = 1.0,
        rollout_sync_interval: int = 8,
        freeze_vision_tower: bool = True,
        log_reward_clip_min: float = -9.21,
        logZ_init: float = 0.0,
        logZ_lr: float = 1e-2,
        logz_hidden_dim: int = 512,
        logz_dropout: float = 0.0,
        logz_max_length: int = 512,
        logz_pooling: str = "mean",
        logz_detach_base_embeddings: bool = True,
        top_p: float = 1.0,
        top_k: int = -1,
        trajectory_log_path: Optional[str] = None,
        trajectory_log_every_steps: int = 50,
        trajectory_log_num_samples: int = 4,
        trajectory_log_include_turn_records: bool = False,
        trajectory_log_max_text_chars: int = 2000,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GFN-TB")

        self.top_p = top_p
        self.top_k = top_k
        self.controller_addr = controller_addr
        self.max_rounds = max_rounds
        self.reward_accuracy_weight = reward_accuracy_weight
        self.reward_format_weight = reward_format_weight
        self.reward_epsilon = reward_epsilon
        self.replay_sampling = replay_sampling
        self.replay_priority_alpha = replay_priority_alpha
        self.rollout_sync_interval = max(1, rollout_sync_interval)
        self.log_reward_clip_min = log_reward_clip_min
        self.logZ_lr = logZ_lr
        self.logz_hidden_dim = logz_hidden_dim
        self.logz_dropout = logz_dropout
        self.logz_max_length = logz_max_length
        self.logz_pooling = logz_pooling
        self.logz_detach_base_embeddings = logz_detach_base_embeddings
        self.replay_buffer = ToolTrajectoryReplayBuffer(
            capacity=replay_buffer_size,
            sampling_mode=replay_sampling,
            priority_alpha=replay_priority_alpha,
            seed=args.seed,
        )
        self._pending_logZ_init = float(logZ_init)
        self._metrics = defaultdict(list)
        self.trajectory_log_path = trajectory_log_path
        self.trajectory_log_every_steps = max(1, trajectory_log_every_steps)
        self.trajectory_log_num_samples = max(1, trajectory_log_num_samples)
        self.trajectory_log_include_turn_records = trajectory_log_include_turn_records
        self.trajectory_log_max_text_chars = trajectory_log_max_text_chars
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["torch_dtype"] = torch.float16

        if isinstance(model, str):
            model_id = model
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            if "Qwen2-VL" in model_id or "Qwen2VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache", None)
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if freeze_vision_tower:
            self._freeze_vision_tower(model)

        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # if not hasattr(model, "gfn_logZ"):
        #     model.register_parameter("gfn_logZ", torch.nn.Parameter(torch.tensor(self._pending_logZ_init, dtype=torch.float32)))
        # self.logZ = model.gfn_logZ

        embed_dim = model.get_input_embeddings().embedding_dim

        if not hasattr(model, "gfn_logz_head"):
            model.add_module(
                "gfn_logz_head",
                PromptConditionedLogZ(
                    embed_dim=embed_dim,
                    hidden_dim=self.logz_hidden_dim,
                    dropout=self.logz_dropout,
                ),
            )

        

        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="left")

        if hasattr(processing_class, "tokenizer"):
            processing_class.tokenizer.truncation_side = "left"
        else:
            processing_class.truncation_side = "left"
        
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = args.use_vllm
        self.model_id = model_id

        def data_collator(features):
            return features

        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.model_accepts_loss_kwargs = False

        if not self.use_vllm:
            raise ValueError("Qwen2VLGFNTBVLLMTrainer requires --use_vllm true")
        if not is_vllm_available():
            raise ImportError("vLLM is required for Qwen2VLGFNTBVLLMTrainer")

        vllm_hf_overrides = self._build_vllm_hf_overrides(model.config)
        if vllm_hf_overrides is not None:
            warnings.warn(
                "Detected a checkpoint config that needs normalization for vLLM "
                "initialization; coercing `text_config` and tied-embedding flags."
            )

        if self.accelerator.is_main_process:
            vllm_device = self.args.vllm_device
            if vllm_device == "auto":
                vllm_device = f"cuda:{self.accelerator.num_processes}"
            if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                raise ValueError(f"Requested vLLM device {vllm_device} is not available")
            if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                warnings.warn(f"The requested device {vllm_device} is also used for training.")
            world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
            profiling_patch = patch(
                "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                return_value=None,
            )
            with world_size_patch, profiling_patch,torch.cuda.device(vllm_device):
                self.llm = LLM(
                    model=model_id,
                    device=vllm_device,
                    gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    dtype=torch.bfloat16,
                    hf_overrides=vllm_hf_overrides,
                    enable_prefix_caching=True,
                    enforce_eager=True,
                    max_model_len=args.max_prompt_length,
                    limit_mm_per_prompt={"image": 6},
                )
            self.sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=self.max_completion_length,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        self._last_loaded_step = -1
        self.accelerator.wait_for_everyone()

    def _freeze_vision_tower(self, model: PreTrainedModel) -> None:
        for name, param in model.named_parameters():
            lowered = name.lower()
            if "vision" in lowered or "visual" in lowered or "image" in lowered:
                param.requires_grad = False


    def _extract_prompt_condition_text(self, prompt):
        if isinstance(prompt, str):
            return prompt

        text_chunks = []
        for msg in prompt:
            content = msg.get("content")
            if isinstance(content, str):
                text_chunks.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_chunks.append(item.get("text", ""))

        return "\n".join(chunk for chunk in text_chunks if chunk)

    def _compute_conditioned_logz_batch(self, model, prompt_texts):
        if not prompt_texts:
            return torch.empty(0, device=model.device, dtype=torch.float32)

        tokenizer = (
            self.processing_class.tokenizer
            if hasattr(self.processing_class, "tokenizer")
            else self.processing_class
        )

        tokenized = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.logz_max_length,
        )

        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)

        base_model = model.module if hasattr(model, "module") else model
        embed_layer = base_model.get_input_embeddings()
        token_embeds = embed_layer(input_ids)

        if self.logz_detach_base_embeddings:
            token_embeds = token_embeds.detach()

        if self.logz_pooling != "mean":
            raise ValueError(f"Unsupported logz_pooling: {self.logz_pooling}")

        mask = attention_mask.unsqueeze(-1).to(token_embeds.dtype)
        summed = (token_embeds * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / denom

        return base_model.gfn_logz_head(pooled).float()

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        decay_parameters = self.get_decay_parameter_names(self.model)
        named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_params
                    if n in decay_parameters and not n.startswith("gfn_logz_head.")
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_params
                    if n not in decay_parameters and not n.startswith("gfn_logz_head.")
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in named_params if n.startswith("gfn_logz_head.")],
                "weight_decay": 0.0,
                "lr": self.logZ_lr,
            },
        ]

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    # def create_optimizer(self):
    #     if self.optimizer is not None:
    #         return self.optimizer

    #     optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    #     decay_parameters = self.get_decay_parameter_names(self.model)
    #     named_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in named_params if n in decay_parameters],
    #             "weight_decay": self.args.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in named_params if n not in decay_parameters],
    #             "weight_decay": 0.0,
    #         },
    #         {
    #             "params": [self.logZ],
    #             "weight_decay": 0.0,
    #             "lr": self.logZ_lr,
    #         },
    #     ]

    #     self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #     return self.optimizer

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _maybe_sync_vllm_weights(self):
        if self.state.global_step == self._last_loaded_step:
            return
        if self.state.global_step > 0 and self.state.global_step % self.rollout_sync_interval != 0 and self._last_loaded_step >= 0:
            return

        # with unwrap_model_for_generation(
        #     self.model,
        #     self.accelerator,
        #     gather_deepspeed3_params=True,
        # ) as unwrapped_model:
        wrapped_model = self.model_wrapped if getattr(self, "model_wrapped", None) is not None else self.model
        with unwrap_model_for_generation(
            wrapped_model,
            self.accelerator,
            gather_deepspeed3_params=is_deepspeed_zero3_enabled(),
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            #llm_model.load_weights(state_dict.items())
            filtered_items = [(k, v) for k, v in state_dict.items() if not k.startswith("gfn_")]
            llm_model.load_weights(filtered_items)
        self._last_loaded_step = self.state.global_step
        self.accelerator.wait_for_everyone()


    # def _maybe_sync_vllm_weights(self):
    #     if self.state.global_step == self._last_loaded_step:
    #         return
    #     if (
    #         self.state.global_step > 0
    #         and self.state.global_step % self.rollout_sync_interval != 0
    #         and self._last_loaded_step >= 0
    #     ):
    #         return

    #     wrapped_model = (
    #         self.model_wrapped
    #         if getattr(self, "model_wrapped", None) is not None
    #         else self.model
    #     )

    #     with unwrap_model_for_generation(
    #         wrapped_model,
    #         self.accelerator,
    #         gather_deepspeed3_params=is_deepspeed_zero3_enabled(),
    #     ) as unwrapped_model:
    #         if is_compiled_module(unwrapped_model):
    #             state_dict = unwrapped_model._orig_mod.state_dict()
    #         else:
    #             state_dict = unwrapped_model.state_dict()

    #     if self.accelerator.is_main_process:
    #         llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
    #         llm_model.load_weights(state_dict.items())

    #     self._last_loaded_step = self.state.global_step
    #     self.accelerator.wait_for_everyone()

    def _truncate_for_log(self, value):
        if isinstance(value, str):
            if len(value) <= self.trajectory_log_max_text_chars:
                return value
            return value[: self.trajectory_log_max_text_chars] + "...<truncated>"
        if isinstance(value, list):
            return [self._truncate_for_log(v) for v in value]
        if isinstance(value, dict):
            return {k: self._truncate_for_log(v) for k, v in value.items()}
        return value

    def _strip_images_from_messages_for_log(self, messages):
        sanitized = copy.deepcopy(messages)

        for msg in sanitized:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue

                item_type = item.get("type")

                if item_type == "image":
                    new_content.append({"type": "image", "image": "<omitted>"})
                    continue

                if item_type == "image_url":
                    new_content.append({"type": "image_url", "image_url": "<omitted>"})
                    continue

                new_content.append(item)

            msg["content"] = new_content

        return sanitized

    def _maybe_log_trajectory_samples(self, prompts, rollout_outputs, replay_items):
        if not self.trajectory_log_path:
            return
        if not self.accelerator.is_main_process:
            return
        if self.state.global_step % self.trajectory_log_every_steps != 0:
            return

        os.makedirs(os.path.dirname(self.trajectory_log_path), exist_ok=True)

        num_samples = min(
            self.trajectory_log_num_samples,
            len(rollout_outputs),
            len(replay_items),
            len(prompts),
        )

        with open(self.trajectory_log_path, "a", encoding="utf-8") as f:
            for i in range(num_samples):
                rollout_item = rollout_outputs[i]
                replay_item = replay_items[i]
                prompt_for_log = prompts[i]
                if isinstance(prompt_for_log, list):
                    prompt_for_log = self._strip_images_from_messages_for_log(prompt_for_log)

                record = {
                    "global_step": int(self.state.global_step),
                    "sample_index": i,
                    "sample_id": replay_item.sample_id,
                    "reward_accuracy": replay_item.reward_accuracy,
                    "reward_format": replay_item.reward_format,
                    "reward_total": replay_item.reward_total,
                    "log_reward": replay_item.log_reward,
                    "num_turns": replay_item.num_turns,
                    "solution": self._truncate_for_log(replay_item.solution),
                    "prompt": self._truncate_for_log(prompt_for_log),
                    "model_outputs": self._truncate_for_log(rollout_item.get("model_outputs", [])),
                    "tool_outputs": self._truncate_for_log(rollout_item.get("tool_outputs", [])),
                    "validation_results": self._truncate_for_log(rollout_item.get("validation_results", [])),
                }

                if self.trajectory_log_include_turn_records:
                    turn_records_for_log = copy.deepcopy(rollout_item.get("turn_records", []))
                    for tr in turn_records_for_log:
                        if "state_messages_before_turn" in tr:
                            tr["state_messages_before_turn"] = self._strip_images_from_messages_for_log(
                                tr["state_messages_before_turn"]
                            )
                    record["turn_records"] = self._truncate_for_log(turn_records_for_log)    

                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
            f.flush()

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompts = [x["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        solutions = [x.get("solution", "") for x in inputs]

        expanded_prompts = []
        expanded_images = []
        expanded_solutions = []
        for prompt, image, solution in zip(prompts, images, solutions):
            for _ in range(self.num_generations):
                expanded_prompts.append(prompt)
                expanded_images.append(image)
                expanded_solutions.append(solution)

        self._maybe_sync_vllm_weights()

        all_prompts = gather_object(expanded_prompts)
        all_images = gather_object(expanded_images)
        all_solutions = gather_object(expanded_solutions)

        if self.accelerator.is_main_process:
            with torch.cuda.device(self.llm.llm_engine.device_config.device):
                rollout_outputs = vllm_generate_with_tool_calls_gfn(
                    self.llm,
                    prompts=all_prompts,
                    images=all_images,
                    sampling_params=self.sampling_params,
                    max_rounds=self.max_rounds,
                    model_mode="general",
                    controller_addr=self.controller_addr,
                )
            replay_items = []
            completion_lengths = []
            tool_counts = []
            for prompt, output_item, solution in zip(all_prompts, rollout_outputs, all_solutions):
                acc = accuracy_reward_from_model_outputs(output_item.get("model_outputs", []), solution)
                fmt = format_reward_from_model_outputs(output_item.get("model_outputs", []))
                reward_total = self.reward_epsilon + self.reward_accuracy_weight * acc + self.reward_format_weight * fmt
                log_reward = math.log(max(reward_total, self.reward_epsilon))
                if math.isfinite(self.log_reward_clip_min):
                    log_reward = max(log_reward, self.log_reward_clip_min)
                replay_items.append(
                    ReplayItem(
                        sample_id=str(uuid.uuid4()),
                        solution=solution,
                        prompt_condition_text=self._extract_prompt_condition_text(prompt),
                        turn_records=output_item.get("turn_records", []),
                        reward_accuracy=float(acc),
                        reward_format=float(fmt),
                        reward_total=float(reward_total),
                        log_reward=float(log_reward),
                        num_turns=len(output_item.get("turn_records", [])),
                    )
                )
                completion_len = sum(len(ids) for ids in output_item.get("model_output_ids", []))
                completion_lengths.append(completion_len)

                tool_num = sum(
                    1
                    for cfg in output_item.get("tool_cfgs", [])
                    if cfg.get("API_name") not in {"Terminate", "ToolValidator", "", None}
                )
                tool_counts.append(tool_num)

            self._maybe_log_trajectory_samples(
                prompts=all_prompts,
                rollout_outputs=rollout_outputs,
                replay_items=replay_items,
            )
                
            self.replay_buffer.extend(replay_items)
            sampled = self.replay_buffer.sample(len(all_prompts)) or replay_items

            # Fresh rollout batch averages
            fresh_reward_total = sum(item.reward_total for item in replay_items) / max(len(replay_items), 1)
            fresh_reward_accuracy = sum(item.reward_accuracy for item in replay_items) / max(len(replay_items), 1)
            fresh_reward_format = sum(item.reward_format for item in replay_items) / max(len(replay_items), 1)
            fresh_num_turns = sum(item.num_turns for item in replay_items) / max(len(replay_items), 1)

            # Sampled replay batch averages
            sampled_reward_total = sum(item.reward_total for item in sampled) / max(len(sampled), 1)
            sampled_reward_accuracy = sum(item.reward_accuracy for item in sampled) / max(len(sampled), 1)
            sampled_reward_format = sum(item.reward_format for item in sampled) / max(len(sampled), 1)
            sampled_num_turns = sum(item.num_turns for item in sampled) / max(len(sampled), 1)

            # Rollout-side diagnostics
            ave_completion_length = sum(completion_lengths) / max(len(completion_lengths), 1)
            ave_tool_num = sum(tool_counts) / max(len(tool_counts), 1)

            # Keep old names for backward compatibility, but make them explicitly sampled
            #self._metrics["reward_total"].append(sampled_reward_total)
            #self._metrics["reward_accuracy"].append(sampled_reward_accuracy)
            #self._metrics["reward_format"].append(sampled_reward_format)
            #self._metrics["num_turns"].append(sampled_num_turns)

            # New explicit metrics
            self._metrics["fresh_reward_total"].append(fresh_reward_total)
            self._metrics["fresh_reward_accuracy"].append(fresh_reward_accuracy)
            self._metrics["fresh_reward_format"].append(fresh_reward_format)
            self._metrics["fresh_num_turns"].append(fresh_num_turns)

            self._metrics["sampled_reward_total"].append(sampled_reward_total)
            self._metrics["sampled_reward_accuracy"].append(sampled_reward_accuracy)
            self._metrics["sampled_reward_format"].append(sampled_reward_format)
            self._metrics["sampled_num_turns"].append(sampled_num_turns)

            self._metrics["replay_size"].append(float(len(self.replay_buffer)))
            self._metrics["completion_length"].append(ave_completion_length)
            self._metrics["avg_tool_num"].append(ave_tool_num)
        else:
            sampled = [None] * len(all_prompts)

        sampled = broadcast_object_list(sampled, from_process=0)
        local_n = len(expanded_prompts)
        process_slice = slice(
            self.accelerator.process_index * local_n,
            (self.accelerator.process_index + 1) * local_n,
        )
        local_replay_items = sampled[process_slice]
        return {"replay_items": local_replay_items}


    def _normalize_messages_for_qwen(self, messages):
        normalized = copy.deepcopy(messages)

        for msg in normalized:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "image_url":
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_value = image_url.get("url")
                    else:
                        image_value = image_url

                    item["type"] = "image"
                    item["image"] = image_value
                    item.pop("image_url", None)

        return normalized

    def _score_turn_logprob(self, model: PreTrainedModel, turn_record: dict[str, Any]) -> torch.Tensor:
        state_messages = self._normalize_messages_for_qwen(
            turn_record["state_messages_before_turn"]
        )
        assistant_action_text = turn_record["assistant_action_text"]

        full_messages = self._normalize_messages_for_qwen(
            [
                *state_messages,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_action_text}],
                },
            ]
        )

        prefix_text = self.processing_class.apply_chat_template(
            state_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.processing_class.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        if not full_text.startswith(prefix_text):
            raise ValueError("full_text does not start with prefix_text; cannot recover assistant suffix reliably")

        suffix_text = full_text[len(prefix_text):]

        full_images, _ = process_vision_info(full_messages)

        full_inputs = self.processing_class(
            text=[full_text],
            images=full_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length + self.max_completion_length,
            add_special_tokens=False,
        )

        tokenizer = (
            self.processing_class.tokenizer
            if hasattr(self.processing_class, "tokenizer")
            else self.processing_class
        )
        suffix_ids = tokenizer(
            suffix_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_completion_length,
            add_special_tokens=False,
        )["input_ids"][0]

        input_ids = full_inputs["input_ids"].to(model.device)
        attention_mask = full_inputs["attention_mask"].to(model.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "pixel_values" in full_inputs:
            model_inputs["pixel_values"] = full_inputs["pixel_values"].to(
                model.device, dtype=torch.bfloat16
            )
        if "image_grid_thw" in full_inputs:
            model_inputs["image_grid_thw"] = full_inputs["image_grid_thw"].to(model.device)

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if model.device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(**model_inputs).logits

        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        suffix_len = min(suffix_ids.shape[0], targets.shape[1])
        if suffix_len == 0:
            return torch.zeros((), device=model.device, dtype=torch.float32)

        logits = logits[:, -suffix_len:, :]
        targets = targets[:, -suffix_len:]

        log_probs = logits.log_softmax(dim=-1)
        token_logps = torch.gather(
            log_probs, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        return token_logps.float().sum().reshape(())

    # def _score_turn_logprob(self, model: PreTrainedModel, turn_record: dict[str, Any]) -> torch.Tensor:
    #     state_messages = self._normalize_messages_for_qwen(
    #         turn_record["state_messages_before_turn"]
    #     )
    #     assistant_action_text = turn_record["assistant_action_text"]

    #     full_messages = self._normalize_messages_for_qwen(
    #         [
    #             *state_messages,
    #             {
    #                 "role": "assistant",
    #                 "content": [{"type": "text", "text": assistant_action_text}],
    #             },
    #         ]
    #     )

    #     prefix_text = self.processing_class.apply_chat_template(
    #         state_messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #     )
    #     full_text = self.processing_class.apply_chat_template(
    #         full_messages,
    #         tokenize=False,
    #         add_generation_prompt=False,
    #     )

    #     prefix_images, _ = process_vision_info(state_messages)
    #     full_images, _ = process_vision_info(full_messages)

    #     prefix_inputs = self.processing_class(
    #         text=[prefix_text],
    #         images=prefix_images,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=self.max_prompt_length,
    #         add_special_tokens=False,
    #     )
    #     full_inputs = self.processing_class(
    #         text=[full_text],
    #         images=full_images,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=self.max_prompt_length,
    #         add_special_tokens=False,
    #     )

        
    #     prefix_len = prefix_inputs["input_ids"].shape[1]
    #     input_ids = full_inputs["input_ids"].to(model.device)
    #     attention_mask = full_inputs["attention_mask"].to(model.device)
    #     model_inputs = {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #     }
    #     if "pixel_values" in full_inputs:
    #         model_inputs["pixel_values"] = full_inputs["pixel_values"].to(model.device, dtype=torch.bfloat16)
    #     if "image_grid_thw" in full_inputs:
    #         model_inputs["image_grid_thw"] = full_inputs["image_grid_thw"].to(model.device)

    #     autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if model.device.type == "cuda" else nullcontext()
    #     with autocast_ctx:
    #         logits = model(**model_inputs).logits

    #     logits = logits[:, :-1, :]
    #     targets = input_ids[:, 1:]
    #     start_idx = max(prefix_len - 1, 0)
    #     logits = logits[:, start_idx:, :]
    #     targets = targets[:, start_idx:]
    #     if logits.size(1) == 0 or targets.numel() == 0:
    #         return torch.zeros((), device=model.device, dtype=torch.float32)

    #     log_probs = logits.log_softmax(dim=-1)
    #     token_logps = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    #     return token_logps.float().sum().reshape(())
        # log_probs = logits.log_softmax(dim=-1)
        # token_logps = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        # return token_logps.sum()



    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     if return_outputs:
    #         raise ValueError("Qwen2VLGFNTBVLLMTrainer does not support return_outputs")
    #     replay_items = inputs["replay_items"]
    #     losses = []
    #     log_pf_values = []
    #     for item in replay_items:
    #         if item is None:
    #             continue
    #         log_pf = torch.zeros((), device=model.device, dtype=torch.float32)
    #         for turn_record in item.turn_records:
    #             log_pf = log_pf + self._score_turn_logprob(model, turn_record).float()
    #         residual = self.logZ + log_pf - torch.tensor(item.log_reward, device=model.device, dtype=torch.float32)
    #         losses.append(residual.pow(2))
    #         log_pf_values.append(log_pf.detach())

    #     if not losses:
    #         loss = self.logZ * 0.0
    #     else:
    #         loss = torch.stack(losses).mean()
    #         mean_log_pf = torch.stack(log_pf_values).mean()
    #         self._metrics["log_pf"].append(mean_log_pf.item())
    #         self._metrics["logZ"].append(self.logZ.detach().float().item())

    #     return loss

    def _as_optional_scalar(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if tensor is None:
            return torch.zeros((), device=device, dtype=torch.float32)

        if not isinstance(tensor, torch.Tensor):
            return torch.tensor(tensor, device=device, dtype=torch.float32).reshape(())

        if tensor.numel() == 0:
            return torch.zeros((), device=device, dtype=torch.float32)

        if tensor.numel() == 1:
            return tensor.float().reshape(())

        raise ValueError(
            f"Expected scalar-like tensor with numel 0 or 1, got shape={tuple(tensor.shape)}"
        )
    
    def _get_logZ_scalar(self, device: torch.device) -> torch.Tensor:
        return self._as_optional_scalar(self.logZ, device=device)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("Qwen2VLGFNTBVLLMTrainer does not support return_outputs")

        replay_items = inputs["replay_items"]
        replay_items = [item for item in replay_items if item is not None]

        if not replay_items:
            return torch.zeros((), device=model.device, dtype=torch.float32, requires_grad=True)

        unique_prompt_texts = list(dict.fromkeys(item.prompt_condition_text for item in replay_items))
        logz_values = self._compute_conditioned_logz_batch(model, unique_prompt_texts)
        prompt_to_logz = {
            prompt_text: logz_value.reshape(())
            for prompt_text, logz_value in zip(unique_prompt_texts, logz_values)
        }

        losses = []
        log_pf_values = []
        logz_batch_values = []

        for item in replay_items:
            log_pf = torch.zeros((), device=model.device, dtype=torch.float32)
            for turn_record in item.turn_records:
                log_pf = log_pf + self._score_turn_logprob(model, turn_record).float().reshape(())

            log_reward = torch.tensor(item.log_reward, device=model.device, dtype=torch.float32).reshape(())
            logz_value = prompt_to_logz[item.prompt_condition_text].reshape(())
            residual = (logz_value + log_pf - log_reward).reshape(())
            loss_i = residual.pow(2).reshape(())

            losses.append(loss_i)
            log_pf_values.append(log_pf.detach())
            logz_batch_values.append(logz_value.detach())

        loss = torch.stack(losses).mean()
        self._metrics["log_pf"].append(torch.stack(log_pf_values).mean().item())
        self._metrics["logZ"].append(torch.stack(logz_batch_values).mean().item())

        return loss
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.accelerator.unwrap_model(self.model)

        if state_dict is None:
            state_dict = model_to_save.state_dict()

        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("gfn_")
        }

        if hasattr(model_to_save, "save_pretrained"):
            model_to_save.save_pretrained(
                output_dir,
                state_dict=filtered_state_dict,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            torch.save(
                filtered_state_dict,
                os.path.join(output_dir, "pytorch_model.bin"),
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        if logs and next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()
