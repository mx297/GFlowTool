
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
from ...utils.chartgemma_metrics import summarize_chartgemma_samples
import os
import json
from datasets import Dataset
from thefuzz import fuzz
try:
    from math_verify import parse, verify
except ImportError:
    parse = None
    verify = None
    print("math_verify package not found. Please install it to use math verification features.")


logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

# def load_dataset(file_path, already_processed_path, num_samples=None):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
#     process_data = set()
#     with open(already_processed_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             process_data.add(data['results']['results']['meta_data']['text'])
#     selected_dataset = []
#     for data in dataset:
#         if data['question'] not in process_data:
#             selected_dataset.append(data)

#     if num_samples is None:
#         return Dataset.from_dict({"data": selected_dataset})
#     return Dataset.from_dict({"data": selected_dataset[:num_samples]})

def load_dataset(file_path, num_samples=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if num_samples is None:
        return Dataset.from_dict({"data": dataset})
    return Dataset.from_dict({"data": dataset[:num_samples]})

def load_data_function():
    
    # raw_data = load_dir_of_jsonl_data_function_default(task_config)
    dataset_path = task_config["dataset_path"]
    image_dir_path = task_config["image_dir_path"]
    num_samples = task_config['num_sample']

    # dataset = load_dataset(dataset_path, already_processed_path, num_samples)
    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    for idx,item in enumerate(dataset):
        item = item["data"]
        item_id = f"chartgemma_{idx}"
        image_file = item.get("image_path")

        image_path = image_file
        text = item["question"]
        label = item.pop("label").replace("<answer> ", "").replace(" </answer>", "").strip()
        data_item = dict(idx=item_id, text=text, label=label, **item)
        data_item["image_path"] = image_path
        meta_data.append(data_item)

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data


# def evaluate_function(results,meta_data):
#     results_dict = {res["idx"]: res for res in results}
#     meta_dict = {meta["idx"]: meta for meta in meta_data}
#     res_list = []
#     compare_logs = []
#     # breakpoint()
#     for idx, meta in meta_dict.items():
#         if idx in results_dict:
#             meta["prediction"] = results_dict[idx]["results"]["final_answer"]
#         else:
#             meta["prediction"] = "None"
        
#         gold = meta["label"].split("<answer>")[-1].split("</answer>")[0].strip().lower()
#         pred = meta["prediction"].lower()
#         gold_latex = parse('${0}$'.format(gold))
#         pred_latex = parse('${0}$'.format(pred))
#         score = 0.0
#         if verify(gold_latex, pred_latex) or gold == pred or verify(gold, pred):
#             score = 1.0
            
#         elif '%' in pred:
#             pred = pred.replace('%','')
#             if '.' in pred:
#                 pred = pred.split('.')[0]
#             if pred == gold or verify(gold, pred):
#                 score = 1.0
#             else:
#                 score = 0.0

#         elif '.' in pred:
#             pred = pred.split('.')[0]
#             if pred == gold or verify(gold, pred):
#                 score = 1.0
#             else:
#                 score = 0.0
#         else:
#             score = fuzz.ratio(pred, gold) / 100
            
#         res_list.append(score)
#         compare_logs.append({"idx":idx,"gold":gold,"pred":pred,"score":score})

#     accuracy = sum(res_list) / len(res_list) if len(res_list) > 0 else 0

#     return {"Acc":accuracy, "compare_logs":compare_logs, "results":results,"meta_data":meta_data}


def _strip_outer_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _normalize_answer(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.strip()

    # Prefer decoding JSON-style string payloads like "\"24.8\"" or "[\"a\", \"b\"]".
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            text = decoded
    except Exception:
        text = _strip_outer_quotes(text)

    return text.strip().lower()


def _numeric_variants(text: str):
    variants = []
    cleaned = text.replace(",", "").strip()

    for candidate in [cleaned, cleaned.replace("%", "").strip()]:
        if not candidate:
            continue
        try:
            variants.append(float(candidate))
        except ValueError:
            pass

    return variants


def _ratio_value(text: str):
    cleaned = text.replace(",", "").strip()
    for sep in [":", "/"]:
        if sep in cleaned:
            parts = cleaned.split(sep, 1)
            try:
                left = float(parts[0].strip())
                right = float(parts[1].strip())
                if right != 0:
                    return left / right
            except ValueError:
                return None
    return None


def _safe_verify(gold: str, pred: str) -> bool:
    if parse is None or verify is None:
        return False
    try:
        gold_latex = parse('${0}$'.format(gold))
        pred_latex = parse('${0}$'.format(pred))
        if verify(gold_latex, pred_latex):
            return True
    except Exception:
        pass

    try:
        if verify(gold, pred):
            return True
    except Exception:
        pass

    return False


def _score_prediction(gold: str, pred: str) -> float:
    if gold == pred or _safe_verify(gold, pred):
        return 1.0

    # Accept missing/extra percent signs when the numeric value is otherwise unchanged.
    gold_nums = _numeric_variants(gold)
    pred_nums = _numeric_variants(pred)
    for g in gold_nums:
        for p in pred_nums:
            if abs(g - p) <= 1e-9:
                return 1.0

    # Accept decimal equivalents for ratio-style gold answers like "35:32" vs "1.09375".
    gold_ratio = _ratio_value(gold)
    pred_ratio = _ratio_value(pred)
    if gold_ratio is not None:
        for p in pred_nums:
            if abs(gold_ratio - p) <= 1e-3:
                return 1.0
    if pred_ratio is not None:
        for g in gold_nums:
            if abs(pred_ratio - g) <= 1e-3:
                return 1.0

    if gold_ratio is not None and pred_ratio is not None and abs(gold_ratio - pred_ratio) <= 1e-3:
        return 1.0

    return fuzz.ratio(pred, gold) / 100


def evaluate_function(results,meta_data):
    results_dict = {res["idx"]: res for res in results}
    metric_config = dict(task_config.get("metric_config", {}))
    correctness_threshold = float(metric_config.get("correctness_threshold", 1.0))

    sample_records = []
    for meta in meta_data:
        idx = meta["idx"]
        result_payload = results_dict.get(idx, {})
        result_item = result_payload.get("results", {})

        prediction = str(result_item.get("final_answer", ""))
        gold = meta["label"].split("<answer>")[-1].split("</answer>")[0].strip().lower()
        pred = _normalize_answer(prediction)
        score = _score_prediction(gold, pred)
        model_responses = result_item.get("model_response", [])

        sample_records.append(
            {
                "idx": idx,
                "prompt_idx": meta.get("prompt_idx", idx),
                "sample_idx": meta.get("sample_idx", 0),
                "num_generations": meta.get("num_generations", 1),
                "question": meta["text"],
                "image_path": meta["image_path"],
                "gold": gold,
                "raw_pred": prediction,
                "pred": pred,
                "score": score,
                "is_correct": float(score) >= correctness_threshold,
                "final_response": model_responses[-1] if model_responses else "",
                "trajectory_text": "\n".join(model_responses),
                "num_rounds": len(model_responses),
                "tool_cfg": result_item.get("tool_cfg", []),
                "tool_response": result_item.get("tool_response", []),
            }
        )

    summary = summarize_chartgemma_samples(sample_records, metric_config)
    summary["Acc"] = summary.get("pass@1")
    summary["task_name"] = task_config["task_name"]
    summary["dataset_path"] = task_config["dataset_path"]
    summary["generation_config"] = dict(task_config.get("generation_config", {}))
    summary["metric_config"] = metric_config
    summary["sample_records"] = sample_records
    return summary
