#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
build_dir=${1:-/tmp/catccos-build}
cmake -S "$repo" -B "$build_dir"
cmake --build "$build_dir" --target matmul_reduce_scatter_shallow_fusion -j4
