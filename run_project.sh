#!/usr/bin/env bash
# Linux / macOS: creates a virtual environment, installs requirements and runs the project.
# Usage: ./run_project.sh            (full run, 20 seeds)
#        ./run_project.sh --seeds 3  (quick demo)
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
python src/zero_trust_prototype.py "$@"
echo "Results are in the folder: outputs_ca2_v2"
