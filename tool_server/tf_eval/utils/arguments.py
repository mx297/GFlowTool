import os

import json
import copy
import random
import logging
import argparse
import numpy as np
from PIL import Image
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, List, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from transformers import TrainerCallback
from transformers import HfArgumentParser, TrainingArguments
from box import Box

from .utils import *

@dataclass
class ModelArguments:
    model: Optional[str] = field(default="qwen2vl")
    model_args: Optional[str] = field(default="pretrained=/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct")
    model_mode: Optional[str] = field(default="opensource")
    batch_size: Optional[int] = field(default=1)
    stop_token: Optional[str] = field(default=None)
    max_rounds: Optional[int] = field(default=3)

@dataclass
class TaskArguments:
    task_name: Optional[str] = field(default="charxiv")
    resume_from_ckpt: Optional[Dict[str, str]] = field(default=None,)
    save_to_ckpt: Optional[Dict[str, str]] = field(default=None,)
    task_config_overrides: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self):
        if self.save_to_ckpt is None:
            self.save_to_ckpt = Box()
        elif isinstance(self.save_to_ckpt, dict):
            self.save_to_ckpt = Box(self.save_to_ckpt)
        else:
            raise ValueError("save_to_ckpt should be a dictionary.")
        if self.resume_from_ckpt is None:
            self.resume_from_ckpt = Box()
        elif isinstance(self.resume_from_ckpt, dict):
            self.resume_from_ckpt = Box(self.resume_from_ckpt)
        else:
            raise ValueError("resume_from_ckpt should be a dictionary.")
        if self.task_config_overrides is None:
            self.task_config_overrides = Box()
        elif isinstance(self.task_config_overrides, dict):
            self.task_config_overrides = Box(self.task_config_overrides)
        else:
            raise ValueError("task_config_overrides should be a dictionary.")


@dataclass
class GenerationArguments:
    num_generations: int = field(default=1)
    max_completion_length: int = field(default=2048)
    do_sample: Optional[bool] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    repetition_penalty: Optional[float] = field(default=None)
    seed: Optional[int] = field(default=None)


@dataclass
class MetricArguments:
    pass_k_list: List[int] = field(default_factory=lambda: [1])
    correctness_threshold: float = field(default=1.0)
    compute_semantic_diversity: bool = field(default=True)
    compute_correct_only_metrics: bool = field(default=True)
    semantic_diversity_model: str = field(
        default="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )
    semantic_diversity_source: str = field(default="trajectory_text")
    include_sample_records_in_summary: bool = field(default=False)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the Evaluation script.
    """
    config: Optional[str] = field(default=None)
    verbosity: Optional[str] = field(default="INFO")
    wandb_args: Optional[str] = field(default="project=mr_eval,entity=mr_eval")
    output_path: Optional[str] = field(default="output")
    sample_output_path: Optional[str] = field(default=None)
    controller_addr: Optional[str] = field(default=None)


def parse_str_into_dict(args_str: str) -> Dict:
    """
    Parse a string of comma-separated key-value pairs into a dictionary.
    """
    args_dict = {}
    for arg in args_str.split(","):
        key, value = arg.split("=")
        args_dict[key] = value
    return args_dict

def parse_str_into_list(args_str: str) -> List:
    """
    Parse a string of comma-separated values into a list.
    """
    # import pdb; pdb.set_trace()
    return args_str.split(",")


def _compact_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _normalize_metric_args(metric_args: MetricArguments) -> MetricArguments:
    if isinstance(metric_args.pass_k_list, str):
        metric_args.pass_k_list = [
            int(value.strip()) for value in metric_args.pass_k_list.split(",") if value.strip()
        ]
    elif isinstance(metric_args.pass_k_list, tuple):
        metric_args.pass_k_list = list(metric_args.pass_k_list)
    return metric_args


def build_args_from_config_item(config_item: Dict[str, Any]) -> Dict[str, Any]:
    model_args = ModelArguments(**config_item.get("model_args", {}))
    task_args = TaskArguments(**config_item.get("task_args", {}))
    generation_args = GenerationArguments(**config_item.get("generation_args", {}))
    metric_args = MetricArguments(**config_item.get("metric_args", {}))
    script_args = ScriptArguments(**config_item.get("script_args", {}))
    return finalize_args(
        model_args=model_args,
        task_args=task_args,
        generation_args=generation_args,
        metric_args=metric_args,
        script_args=script_args,
        config=config_item,
    )


def finalize_args(
    model_args: ModelArguments,
    task_args: TaskArguments,
    generation_args: GenerationArguments,
    metric_args: MetricArguments,
    script_args: ScriptArguments,
    config=None,
) -> Dict[str, Any]:
    script_args.config = config
    task_args.task_name = parse_str_into_list(task_args.task_name)
    if isinstance(model_args.model_args, str):
        model_args.model_args = parse_str_into_dict(model_args.model_args)
    if isinstance(script_args.wandb_args, str):
        script_args.wandb_args = parse_str_into_dict(script_args.wandb_args)
    metric_args = _normalize_metric_args(metric_args)
    return dict(
        model_args=model_args,
        task_args=task_args,
        generation_args=generation_args,
        metric_args=metric_args,
        script_args=script_args,
    )

def parse_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TaskArguments, GenerationArguments, MetricArguments, ScriptArguments))
    # breakpoint()
    model_args, task_args, generation_args, metric_args, script_args = parser.parse_args_into_dataclasses()
    
    if script_args.config:
        if script_args.config.endswith(".json"):
            config = load_json_file(script_args.config)
        elif script_args.config.endswith(".yaml"):
            config = load_yaml_file(script_args.config)
        else:
            raise ValueError("Config file should be either a json or yaml file.")
        
        if isinstance(config, dict):
            return build_args_from_config_item(config)
        elif isinstance(config, list):
            return build_args_from_config_item(config[0])
        else:
            raise ValueError("Config file should be either a dict or list of dicts.")

    return finalize_args(
        model_args=model_args,
        task_args=task_args,
        generation_args=generation_args,
        metric_args=metric_args,
        script_args=script_args,
        config=None,
    )
    
