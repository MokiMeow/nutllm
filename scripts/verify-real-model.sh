#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 STORIES15M_BIN TOKENIZER_BIN" >&2
    exit 2
fi

model=$1
tokenizer=$2
expected_model=cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a
expected_tokenizer=50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361

actual_model=$(sha256sum "$model" | cut -d' ' -f1)
actual_tokenizer=$(sha256sum "$tokenizer" | cut -d' ' -f1)
if [[ $actual_model != "$expected_model" ]]; then
    echo "unexpected model sha256: $actual_model" >&2
    exit 1
fi
if [[ $actual_tokenizer != "$expected_tokenizer" ]]; then
    echo "unexpected tokenizer sha256: $actual_tokenizer" >&2
    exit 1
fi

make clean
make all
NUTLLM_REAL_MODEL="$model" NUTLLM_REAL_TOKENIZER="$tokenizer" \
    ./build/nutllm 128 | tee build/real-model-proof.log
grep -q '^  ok external 15M checkpoint' build/real-model-proof.log
grep -q 'Once upon a time, there was a little girl named Lily' \
    build/real-model-proof.log

output=$(./build/nutllm --model "$model" --tokenizer "$tokenizer" \
    --prompt "Once upon a time" --max-tokens 16)
case $output in
    "Once upon a time, there was a little girl named Lily."*) ;;
    *)
        echo "unexpected greedy output: $output" >&2
        exit 1
        ;;
esac
echo "[ok] real-model proof"
