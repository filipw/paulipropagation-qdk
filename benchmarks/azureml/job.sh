#!/usr/bin/env bash
# Runs PauliPropagation.jl, this package and cuPauliProp on the same benchmark circuits, on one
# machine with an NVIDIA GPU.
#
#   bash benchmarks/azureml/job.sh <folder for the results> <case> [case ...]
#
# Expects a python with paulipropagation-qdk, cuquantum and cupy. Julia and PauliPropagation.jl
# are installed into a temporary folder. Every part writes its results to the folder case by
# case, so a run that is cut short keeps what it had. A part that fails does not stop the
# others, and the script fails at the end if any did.
set -uo pipefail

JULIA_VERSION=1.13.3
JULIA_SHA256=5c61cb4b40332cb5a3504e7779e2f5dac093caa4a1116ded19f01413f85c63e4
PAULIPROPAGATION_VERSION=0.9.2

mkdir -p "$1"
RESULTS=$(cd "$1" && pwd)
shift
CASES=("$@")
CORES=$(nproc)
START=$(date +%s)
FAILED=()

# The benchmarks write next to themselves, so they run in a copy whose results are the folder.
WORK=$(mktemp -d)
cp -r "$(cd "$(dirname "$0")/../.." && pwd)" "$WORK/repository"
cd "$WORK/repository" || exit 1
rm -rf benchmarks/results
ln -s "$RESULTS" benchmarks/results

part() {
    local name=$1
    shift
    echo "=== $name"
    if ! "$@" 2>&1 | tee "$RESULTS/$name.log"; then
        FAILED+=("$name")
    fi
}

machine() {
    lscpu | grep -E "^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)):"
    free -g | head -2
    nvidia-smi
    python --version
    python -m pip list 2>/dev/null | grep -i -E "^(paulipropagation|cuquantum|cupauliprop|cupy|numpy)" || true
}

julia_benchmark() {
    local julia="$WORK/julia-$JULIA_VERSION/bin/julia"
    local url="https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-$JULIA_VERSION-linux-x86_64.tar.gz"
    export JULIA_DEPOT_PATH="$WORK/depot"
    python -c "import sys, urllib.request; urllib.request.urlretrieve(*sys.argv[1:])" "$url" "$WORK/julia.tar.gz" &&
        echo "$JULIA_SHA256  $WORK/julia.tar.gz" | sha256sum --check - &&
        tar -xzf "$WORK/julia.tar.gz" -C "$WORK" &&
        "$julia" --project="$WORK/environment" -e "
            using Pkg
            Pkg.add([PackageSpec(name=\"PauliPropagation\", version=\"$PAULIPROPAGATION_VERSION\"), PackageSpec(name=\"JSON\")])" &&
        "$julia" -t "$CORES" --project="$WORK/environment" tools/julia/benchmark.jl "${CASES[@]}"
}

THREADS="1,$CORES"
if [ "$CORES" -gt 8 ]; then THREADS="1,8,$CORES"; fi

part machine machine
part julia julia_benchmark
part paulipropagation_qdk python benchmarks/run.py --threads "$THREADS" "${CASES[@]}"
part cupauliprop python benchmarks/cupauliprop.py --threads "$CORES" "${CASES[@]}"

python - "$RESULTS/job.json" "$(($(date +%s) - START))" ${FAILED[@]+"${FAILED[@]}"} <<'PYTHON'
import json, sys
json.dump({"seconds": int(sys.argv[2]), "failed": sys.argv[3:]}, open(sys.argv[1], "w"))
PYTHON
echo "=== $(($(date +%s) - START)) s, failed: ${FAILED[*]-nothing}"
[ ${#FAILED[@]} -eq 0 ]
