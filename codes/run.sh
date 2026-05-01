#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -v "$project_root":/work -w /work/codes \
  ghcr.io/fenics/dolfinx/dolfinx:v0.10.0-r1 \
  python3 solve_cat_heat.py "$@"
