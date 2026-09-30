#!/usr/bin/env bash
# Run with `source ./setup.sh` to keep the environment active in your shell.
_numpycal_setup() {
  local project_dir
  if [ -n "${ZSH_VERSION:-}" ]; then
    project_dir="${(%):-%x}"
  else
    project_dir="${BASH_SOURCE[0]}"
  fi
  project_dir="$(cd "$(dirname "$project_dir")" && pwd)"
  command -v conda >/dev/null 2>&1 || { echo "Install Miniconda first."; return 1; }
  source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
  if ! conda run -n numpycal python -c 'import sys; assert sys.version_info[:2] == (3, 11); import numpy, flask' >/dev/null 2>&1; then
    conda env create -f "$project_dir/environment.yml" || return 1
  fi
  conda activate numpycal || return 1
  echo "Ready. Run: python $project_dir/app.py"
  echo "If you ran bash setup.sh, activate in your shell with: conda activate numpycal"
}
_numpycal_setup
