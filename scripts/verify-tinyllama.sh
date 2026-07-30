#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ $# -ne 2 ]]; then
    echo "usage: $0 TINYLLAMA_BIN TOKENIZER_BIN" >&2
    exit 2
fi

model=$1
tokenizer=$2
timeout_seconds=${NUTLLM_MODEL_TIMEOUT:-600}
expected_model=6d12ab6e18a5c1216c16053fc20647a6438fca483e4586271306e13b082213f4
expected_tokenizer=e610c22d05d092569bafcc2e3f9795b9b43c829fab53d3489d454614cc5b87ce

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
for quant in int8 int4; do
    log="build/tinyllama-${quant}.log"
    timeout "$timeout_seconds" ./build/nutllm \
        --model "$model" --tokenizer "$tokenizer" \
        --prompt "Once upon a time" --max-tokens 16 \
        --quant "$quant" --threads 4 --stats 2>&1 | tee "$log"
    grep -q "^stats format=${quant} threads=4 " "$log"
    grep -q '^Once upon a time' "$log"
done
grep -q 'What is the capital of France?' build/tinyllama-int8.log
grep -q 'Dame Fortune' build/tinyllama-int4.log
echo "[ok] TinyLlama INT8/INT4 proof"
