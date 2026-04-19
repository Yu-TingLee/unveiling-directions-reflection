#!/usr/bin/env bash
set -euo pipefail
source venv/bin/activate

export CUDA_VISIBLE_DEVICES="${1:-0}" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
THREAD="${CUDA_VISIBLE_DEVICES}"

# config
MODELS=("Qwen2.5-3B" "Gemma-3-4B-it")
declare -A HF_ID=(
    [Qwen2.5-3B]="Qwen/Qwen2.5-3B"
    [Gemma-3-4B-it]="google/gemma-3-4b-it"
)
declare -A LAYERS=(                        # steering candidate layers per model
    [Qwen2.5-3B]="$(seq -s ' ' 0 35)"
    [Gemma-3-4B-it]="$(seq -s ' ' 0 33)"
)
declare -A WAIT_L1=(                       # model-specific "level-1" reflection tokens
    [Qwen2.5-3B]="<|endoftext|> % #"
    [Gemma-3-4B-it]="<eos> % #"
)
WAIT_L2="Wait Alternatively Check"
WAIT_L0="Answer Result Output"

TASKS=("gsm8k_adv" "cruxeval_o_adv")
declare -A TASK_LIMIT=(
    [gsm8k_adv]=2000
    [cruxeval_o_adv]=500
)

REFLECTION_WORDS=("Wait" "Alternatively" "Check" "<|endoftext|>" "#" "%" "Answer" "Output" "Result")

# Suffix ab = μ_{La -> Lb}. POS = +μ (induce reflection); NEG = -μ (suppress). Step 5 sweeps one layer per run.
POS_WORDS=("<|endoftext|>" "Answer" "#" "%" "Output" "Result");     POS_SUFFIXES=("20" "21")
NEG_WORDS=("Wait" "<|endoftext|>" "Alternatively" "Check" "#" "%"); NEG_SUFFIXES=("20" "10")

S_LIMIT=200          # build_vectors sample count
WORD_LIMIT=5         # filter_words cap
VIZ_DIR="visualize"
DATA_DIR="mydataset"
TASK_DIR="mytasks"

# path templates
viz_model()  { echo "${VIZ_DIR}/$1/$2"; }                   # task, model
step2_dir()  { echo "$(viz_model "$1" "$2")/step2/$3"; }    # task, model, steer_type
gt_out()     { echo "$(viz_model "$1" "$2")/$3/gt_$4_$5"; } # task, model, step, limit, word
steer_out()  { echo "$(viz_model "$1" "$2")/step4/s$3_$4_$5_$6_tselected_l$7"; } # task, model, scale, word, limit, suffix, layer

# lm_eval wrapper 
# Writes the wait-token file, then runs lm_eval via steered_model.py.
# Extra model_args (e.g. steering_vec_file=...) go in $5 and are comma-appended.

lmeval() {
    local task=$1 model=$2 word=$3 out=$4 extra_args="${5:-}"
    [ -d "${out}" ] && return 0

    echo "${word}" > "${TASK_DIR}/${task}/wait_token_${THREAD}.txt"

    local args="pretrained=${HF_ID[${model}]},dtype=float,trust_remote_code=True"
    [ -n "${extra_args}" ] && args="${args},${extra_args}"

    python3 steered_model.py --model steered_hf \
        --model_args   "${args}" \
        --include_path "${TASK_DIR}" \
        --tasks        "${task}_${THREAD}" \
        --output_path  "${out}" \
        --limit        "${TASK_LIMIT[${task}]}" \
        --device       cuda:0 \
        --batch_size   auto:1 \
        --gen_kwargs   max_new_tokens=256 \
        --seed 0 --log_samples
}

# step 1: preprocess

for task in "${TASKS[@]}"; do
    python3 preprocess.py \
        --input_file="${DATA_DIR}/${task}/train.json" \
        --json_out_name="${task}.json" \
        --visualize_dir="${VIZ_DIR}"
done

# step 2: measure baseline accuracy

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        for word in "${REFLECTION_WORDS[@]}"; do
            lmeval "${task}" "${model}" "${word}" \
                   "$(gt_out "${task}" "${model}" step1 "${TASK_LIMIT[${task}]}" "${word}")"
        done
    done
done

# step 3: build steering vectors + filter candidate words

for model in "${MODELS[@]}"; do
    hf_id="${HF_ID[${model}]}"
    wait_l1="${WAIT_L1[${model}]}"

    for task in "${TASKS[@]}"; do
        src="${VIZ_DIR}/${task}/step0/${task}.json"
        vecs_base="$(viz_model "${task}" "${model}")/step2"

        # (wait_token_1, wait_token_2) pairs define which "reflection level" the direction runs between.
        declare -a PAIRS=(
            "steer_${S_LIMIT}_20 | ${WAIT_L2} | ${WAIT_L0}"
            "steer_${S_LIMIT}_21 | ${WAIT_L2} | ${wait_l1}"
            "steer_${S_LIMIT}_10 | ${wait_l1}  | ${WAIT_L0}"
        )
        for p in "${PAIRS[@]}"; do
            IFS='|' read -r name w1 w2 <<< "${p}"
            python3 build_vectors.py \
                --input_file="${src}" --model_name="${hf_id}" \
                --output_dir="${vecs_base}/${name// /}" --limit "${S_LIMIT}" \
                --wait_token_1 ${w1} --wait_token_2 ${w2}
        done
        python3 build_vectors.py \
            --input_file="${src}" --model_name="${hf_id}" \
            --output_dir="${vecs_base}/steer_baseline" --limit "${S_LIMIT}" \
            --is_baseline=1 --output_new_vec 0 \
            --wait_token_1 ${WAIT_L2} --wait_token_2 ""

        for steer_type in steer_${S_LIMIT}_21 steer_${S_LIMIT}_20 steer_${S_LIMIT}_10 steer_baseline; do
            python3 filter_words.py --input_dir="${vecs_base}/${steer_type}" --word_limit "${WORD_LIMIT}"
        done
    done
done

# step 4: instruction selection (eval top-ranked words from each layer)

eval_words_from_file() {
    local word_file=$1 model=$2 max_words=${3:-0}
    [ -f "${word_file}" ] || return 0
    local i=0
    while IFS= read -r word; do
        [ -z "${word}" ] && continue
        [ "${max_words}" -gt 0 ] && [ "${i}" -ge "${max_words}" ] && break
        lmeval "gsm8k_adv" "${model}" "${word}" \
               "$(gt_out gsm8k_adv "${model}" step3 "${TASK_LIMIT[gsm8k_adv]}" "${word}")"
        i=$((i + 1))
    done < "${word_file}"
}

for model in "${MODELS[@]}"; do
    read -ra layers <<< "${LAYERS[${model}]}"
    base="${VIZ_DIR}/gsm8k_adv/${model}/step2"

    eval_words_from_file "${base}/steer_baseline/word_-1.txt" "${model}"
    for steer_type in steer_${S_LIMIT}_21 steer_${S_LIMIT}_20; do
        for layer in "${layers[@]}"; do
            eval_words_from_file "${base}/${steer_type}/word_${layer}.txt" "${model}" 8
        done
    done
done

# step 5: activation steering (apply vectors at each layer)

run_steer() {
    local model=$1 task=$2 scale=$3 word=$4 suffix=$5 layer=$6
    local steer_file="$(step2_dir "${task}" "${model}" "steer_${S_LIMIT}_${suffix}")/seed_avg.json"
    local args="steering_vec_file=${steer_file},control_type=selected,control_num=${layer},control_scale=${scale}"
    local out="$(steer_out "${task}" "${model}" "${scale}" "${word}" "${TASK_LIMIT[${task}]}" "${suffix}" "${layer}")"
    lmeval "${task}" "${model}" "${word}" "${out}" "${args}"
}

for model in "${MODELS[@]}"; do
    read -ra layers <<< "${LAYERS[${model}]}"
    for task in "${TASKS[@]}"; do
        for layer in "${layers[@]}"; do
            for word in "${POS_WORDS[@]}"; do
                for suffix in "${POS_SUFFIXES[@]}"; do run_steer "${model}" "${task}"  1 "${word}" "${suffix}" "${layer}"; done
            done
            for word in "${NEG_WORDS[@]}"; do
                for suffix in "${NEG_SUFFIXES[@]}"; do run_steer "${model}" "${task}" -1 "${word}" "${suffix}" "${layer}"; done
            done
        done
    done
done

# step 6: plot

python3 plot.py \
    --visualize_dir "${VIZ_DIR}" \
    --model_names   "${MODELS[@]}" \
    --dataset_names "${TASKS[@]}" \
    --s_limit       "${S_LIMIT}" \
    --limits        "${TASK_LIMIT[gsm8k_adv]}" "${TASK_LIMIT[cruxeval_o_adv]}"
