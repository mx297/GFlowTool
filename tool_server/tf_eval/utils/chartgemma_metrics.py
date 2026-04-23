import math
import re
from collections import defaultdict

import numpy as np
from thefuzz import fuzz

try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None


def normalize_answer(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip()).casefold()
    normalized = (
        normalized.replace("—", "-")
        .replace("–", "-")
        .replace("−", "-")
        .replace("’", "'")
        .replace("`", "'")
    )
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("%", "")
    return normalized.strip().strip("\"'")


def compute_chartgemma_score(gold: str, pred: str) -> float:
    gold = normalize_answer(gold)
    pred = normalize_answer(pred)
    if not pred:
        return 0.0

    if gold == pred:
        return 1.0

    if parse is not None and verify is not None:
        try:
            gold_latex = parse(f"${gold}$")
            pred_latex = parse(f"${pred}$")
            if verify(gold_latex, pred_latex) or verify(gold, pred):
                return 1.0
        except Exception:
            pass

    if pred.endswith("%"):
        pred = pred[:-1].strip()
    if "." in pred:
        integer_pred = pred.split(".", 1)[0]
        if integer_pred == gold:
            return 1.0
        if parse is not None and verify is not None:
            try:
                if verify(gold, integer_pred):
                    return 1.0
            except Exception:
                pass

    return fuzz.ratio(pred, gold) / 100.0


def unbiased_pass_at_k(n: int, c: int, k: int):
    if k <= 0 or n <= 0 or n < k:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


_EMBEDDER_CACHE = {}


def get_embedder(model_name: str):
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def mean_pairwise_cosine_distance(embeddings: np.ndarray):
    if len(embeddings) < 2:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = embeddings / norms
    similarity = normalized @ normalized.T
    upper = np.triu_indices(len(embeddings), k=1)
    distances = 1.0 - similarity[upper]
    return float(np.mean(distances))


def compute_semantic_diversity(texts, model_name: str):
    valid_texts = [text for text in texts if str(text).strip()]
    if len(valid_texts) < 2:
        return None
    embedder = get_embedder(model_name)
    embeddings = embedder.encode(valid_texts, convert_to_numpy=True)
    return mean_pairwise_cosine_distance(embeddings)


def summarize_chartgemma_samples(sample_records, metric_config):
    pass_k_list = sorted(set(metric_config.get("pass_k_list", [1])))
    correctness_threshold = float(metric_config.get("correctness_threshold", 1.0))
    compute_semantic = bool(metric_config.get("compute_semantic_diversity", True))
    semantic_model = metric_config.get(
        "semantic_diversity_model",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
    )
    semantic_source = metric_config.get("semantic_diversity_source", "trajectory_text")
    compute_correct_only = bool(metric_config.get("compute_correct_only_metrics", True))

    grouped = defaultdict(list)
    for record in sample_records:
        grouped[record["prompt_idx"]].append(record)

    prompt_metrics = []
    pass_at_k_values = {k: [] for k in pass_k_list}
    semantic_values = []
    semantic_correct_only_values = []
    semantic_error = None

    for prompt_idx, records in grouped.items():
        records = sorted(records, key=lambda item: item["sample_idx"])
        n = len(records)
        c = sum(float(record["score"]) >= correctness_threshold for record in records)
        prompt_summary = {
            "prompt_idx": prompt_idx,
            "num_samples": n,
            "num_correct": c,
        }

        for k in pass_k_list:
            value = unbiased_pass_at_k(n=n, c=c, k=k)
            prompt_summary[f"pass@{k}"] = value
            if value is not None:
                pass_at_k_values[k].append(value)

        if compute_semantic:
            texts = [record.get(semantic_source, "") for record in records]
            try:
                semantic_value = compute_semantic_diversity(texts, semantic_model)
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
                semantic_value = None
            prompt_summary["semantic_diversity"] = semantic_value
            if semantic_value is not None:
                semantic_values.append(semantic_value)

            if compute_correct_only:
                correct_texts = [
                    record.get(semantic_source, "")
                    for record in records
                    if float(record["score"]) >= correctness_threshold
                ]
                try:
                    semantic_correct = compute_semantic_diversity(correct_texts, semantic_model)
                except Exception as exc:
                    semantic_error = f"{type(exc).__name__}: {exc}"
                    semantic_correct = None
                prompt_summary["semantic_diversity_correct_only"] = semantic_correct
                if semantic_correct is not None:
                    semantic_correct_only_values.append(semantic_correct)

        prompt_metrics.append(prompt_summary)

    summary = {
        "num_prompts": len(grouped),
        "num_samples": len(sample_records),
    }
    for k in pass_k_list:
        key = f"pass@{k}"
        values = pass_at_k_values[k]
        summary[key] = float(np.mean(values)) if values else None

    target_n = max((record.get("num_generations", 1) for record in sample_records), default=1)
    if compute_semantic:
        summary[f"semantic_diversity@{target_n}"] = (
            float(np.mean(semantic_values)) if semantic_values else None
        )
        if compute_correct_only:
            summary[f"semantic_diversity@{target_n}_correct_only"] = (
                float(np.mean(semantic_correct_only_values))
                if semantic_correct_only_values
                else None
            )
        summary["semantic_diversity_model"] = semantic_model
        summary["semantic_diversity_source"] = semantic_source
        if semantic_error:
            summary["semantic_diversity_error"] = semantic_error

    summary["coverage"] = {
        "prompts_with_semantic_diversity": len(semantic_values),
        "prompts_with_semantic_diversity_correct_only": len(semantic_correct_only_values),
    }
    summary["prompt_metrics"] = prompt_metrics

    return summary
