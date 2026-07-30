#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ $# -ne 4 ]]; then
    echo "usage: $0 TINYLLAMA_BIN TOKENIZER_BIN LLAMA_BENCH Q4_0_GGUF" >&2
    exit 2
fi

model=$1
tokenizer=$2
llama_bench=$3
gguf=$4
timeout_seconds=${NUTLLM_MODEL_TIMEOUT:-600}
expected_model=6d12ab6e18a5c1216c16053fc20647a6438fca483e4586271306e13b082213f4
expected_tokenizer=e610c22d05d092569bafcc2e3f9795b9b43c829fab53d3489d454614cc5b87ce
expected_gguf=3849e8024b234f2ec0f2e3b5b59ea368804486563394886ae03d0a67ae70d504

check_hash() {
    local path=$1
    local expected=$2
    local actual
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    if [[ $actual != "$expected" ]]; then
        echo "unexpected sha256 for $path: $actual" >&2
        exit 1
    fi
}

check_hash "$model" "$expected_model"
check_hash "$tokenizer" "$expected_tokenizer"
check_hash "$gguf" "$expected_gguf"
test -x "$llama_bench"

make clean
make all
for threads in 1 4; do
    timeout "$timeout_seconds" ./build/nutllm \
        --model "$model" --tokenizer "$tokenizer" \
        --prompt "Once upon a time" --max-tokens 16 \
        --quant int4 --threads "$threads" --stats 2>&1 |
        tee "build/nutllm-int4-t${threads}.log"
done

timeout 600 "$llama_bench" \
    -m "$gguf" -p 5 -n 16 -t 1,4 -r 5 -ngl 0 |
    tee build/llama-bench-q4_0.log
echo "[ok] TinyLlama comparison captured under build/"
