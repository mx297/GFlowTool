#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export WANDB_PROJECT="OpenThinkIMG-GFlowNet"
export WANDB_ENTITY='matef297-mbzuai'
export WANDB_NAME="tool-tbgfn-uniform-temp0.4-top_p_0.95-top_k20_lr5e-6"
RUN_NAME="${WANDB_NAME}"
# -------- User-editable variables --------
MODEL_NAME_OR_PATH="/share_5/users/mohamed_atef/OpenThinkIMG/output/Qwen2.5-VL_sft"   # Base vision-language model to fine-tune.
DATASET_JSON="/share_5/users/mohamed_atef/OpenThinkIMG/tool_dataset/records.jsonl"                 # JSON dataset path passed to --dataset_name.
OUTPUT_DIR="/share_5/users/mohamed_atef/OpenThinkIMG/output/tool-tbgfn-uniform-temp0.4-top_p_0.95-top_k20_lr5e-6"                 # Directory where checkpoints/logs are written.
DEEPSPEED_CONFIG="/share_5/users/mohamed_atef/OpenThinkIMG/r1_v/open_r1/deepspeed_zero2_tool_tbgfn.json"  # DeepSpeed ZeRO-2 config file.

FINETUNE_MODE="full"             # lora = adapter tuning, full = full fine-tuning.
FREEZE_VISION_TOWER="true"       # Freeze vision encoder parameters to save VRAM.
MAX_ROUNDS=6                      # Maximum number of tool-calling turns per rollout.
NUM_GENERATIONS=2                 # Number of rollout samples generated per prompt.
TEMPERATURE=0.4     # Controls randomness of token sampling.
TOP_P=0.9          # Nucleus sampling threshold.
TOP_K=20            # Keep only top-k candidate tokens at each step.

REWARD_ACCURACY_WEIGHT=3.0        # Weight on final-answer accuracy reward.
REWARD_FORMAT_WEIGHT=0.25         # Weight on strict-format reward.
REWARD_EPSILON=1e-4               # Small positive floor added before taking log reward.

REPLAY_BUFFER_SIZE=1000           # Number of trajectories stored in replay.
REPLAY_SAMPLING="prioritized"    # prioritized or uniform replay sampling.
REPLAY_PRIORITY_ALPHA=0.0        # Priority exponent; 0.0 makes prioritized behave like uniform.
ROLLOUT_SYNC_INTERVAL=1           # Sync HF training weights into vLLM every N optimizer steps.

PER_DEVICE_TRAIN_BATCH_SIZE=1     # Micro-batch size per GPU for training.
GRADIENT_ACCUMULATION_STEPS=8     # Number of micro-batches accumulated before an optimizer step.
LEARNING_RATE=5e-6                # Main model learning rate.
LOGZ_LR=1e-3                     # Separate learning rate for scalar logZ.
LOGZ_HIDDEN_DIM=512              # Hidden size of the prompt-conditioned logZ MLP.
LOGZ_DROPOUT=0.0                 # Dropout inside the logZ head.
LOGZ_MAX_LENGTH=512              # Max prompt-text tokens used to condition logZ.
LOGZ_POOLING="mean"              # How prompt token embeddings are pooled; use "mean".
LOGZ_DETACH_BASE_EMBEDDINGS="true"  # If true, logZ head does not backprop into base token embeddings.
MIN_PIXELS=3136                    # Keep low to avoid needless upscaling of already-small charts.
MAX_PIXELS=524288                 # Balanced cap: saves memory while preserving most chart resolution.
MAX_PROMPT_LENGTH=8192            # Max prompt tokens for the vLLM/HF processor side.
MAX_COMPLETION_LENGTH=1024         # Max generated assistant tokens across a rollout.
VLLM_DEVICE="cuda:2"               # Dedicated vLLM device. auto = next visible GPU after training ranks.
VLLM_GPU_MEMORY_UTILIZATION=0.50  # Fraction of the vLLM GPU reserved for inference engine allocations.

NPROC_PER_NODE=2               # Number of training processes / GPUs on this node.
MASTER_PORT=29521                 # Distributed rendezvous port.
SEED=42                           # Random seed.
CONTROLLER_ADDR="http://localhostcodex:20101"  # Explicit tool controller endpoint to avoid implicit fallback.

TRAJECTORY_LOG_PATH="/share_5/users/mohamed_atef/OpenThinkIMG/output/tool-tbgfn-uniform-temp0.4-top_p_0.95-top_k20_lr5e-6/output.jsonl"   # JSONL file for sampled rollouts.
TRAJECTORY_LOG_EVERY_STEPS=1                                          # Log samples every N optimizer steps.
TRAJECTORY_LOG_NUM_SAMPLES=2                                           # Number of sampled trajectories to save each time.
TRAJECTORY_LOG_INCLUDE_TURN_RECORDS="false"                            # Whether to include full per-turn state snapshots.
TRAJECTORY_LOG_MAX_TEXT_CHARS=8000                                     # Truncate long text fields before writing.

# ---------------------------------------
#--max_step 10 \

torchrun \
  --standalone \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_port "${MASTER_PORT}" \
  -m r1_v.open_r1.tool_gfn_tb \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --dataset_name "${DATASET_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --finetune_mode "${FINETUNE_MODE}" \
  --freeze_vision_tower "${FREEZE_VISION_TOWER}" \
  --max_rounds "${MAX_ROUNDS}" \
  --num_generations "${NUM_GENERATIONS}" \
  --reward_accuracy_weight "${REWARD_ACCURACY_WEIGHT}" \
  --reward_format_weight "${REWARD_FORMAT_WEIGHT}" \
  --reward_epsilon "${REWARD_EPSILON}" \
  --replay_buffer_size "${REPLAY_BUFFER_SIZE}" \
  --replay_sampling "${REPLAY_SAMPLING}" \
  --replay_priority_alpha "${REPLAY_PRIORITY_ALPHA}" \
  --rollout_sync_interval "${ROLLOUT_SYNC_INTERVAL}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0 \
  --logZ_lr "${LOGZ_LR}" \
  --min_pixels "${MIN_PIXELS}" \
  --max_pixels "${MAX_PIXELS}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH}" \
  --vllm_device "${VLLM_DEVICE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --use_vllm true \
  --bf16 true \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --logging_steps 1 \
  --save_steps 100 \
  --save_total_limit 3 \
  --num_train_epochs 1 \
  --report_to wandb \
  --run_name "${RUN_NAME}" \
  --max_grad_norm 5 \
  --seed "${SEED}" \
  --controller_addr "${CONTROLLER_ADDR}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --trajectory_log_path "${TRAJECTORY_LOG_PATH}" \
  --trajectory_log_every_steps "${TRAJECTORY_LOG_EVERY_STEPS}" \
  --trajectory_log_num_samples "${TRAJECTORY_LOG_NUM_SAMPLES}" \
  --trajectory_log_include_turn_records "${TRAJECTORY_LOG_INCLUDE_TURN_RECORDS}" \
  --trajectory_log_max_text_chars "${TRAJECTORY_LOG_MAX_TEXT_CHARS}" \
  --logz_hidden_dim "${LOGZ_HIDDEN_DIM}" \
  --logz_dropout "${LOGZ_DROPOUT}" \
  --logz_max_length "${LOGZ_MAX_LENGTH}" \
  --logz_pooling "${LOGZ_POOLING}" \
  --logz_detach_base_embeddings "${LOGZ_DETACH_BASE_EMBEDDINGS}" \
