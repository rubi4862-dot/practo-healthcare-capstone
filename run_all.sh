#!/usr/bin/env bash
# run_all.sh — run every part under MOCK_LLM, fully offline, and save transcripts.
# Prerequisite (once, needs network):  pip install -r requirements.txt && python prefetch_model.py
set -euo pipefail
cd "$(dirname "$0")"

export MOCK_LLM=1
export CREWAI_DISABLE_TELEMETRY=true CREWAI_TRACING_ENABLED=false OTEL_SDK_DISABLED=true
export ANONYMIZED_TELEMETRY=False HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # proves zero network at run time
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export PYTHONUTF8=1                                 # Windows: CrewAI logs emoji; avoid cp1252 errors

mkdir -p transcripts
run() {
  local name=$1; shift
  echo ">>> $name"
  { echo "\$ MOCK_LLM=1 $*"; echo; python "$@"; } 2>&1 | tee "transcripts/$name.txt"
}

run 00_dataset          dataset.py
run 01_part1_rag        part1_rag.py          # must run first: writes results/calibration.json
run 02_part2_status_tool part2_status_tool.py
run 03_part2_crew       part2_crew.py
run 04_part2_memory     part2_memory.py
run 05_part2_guardrails part2_guardrails.py
run 06_part3_api        part3_api_demo.py
run 07_part3_eval       part3_eval.py
run 08_part4_autogen    part4_autogen.py
run 09_part4_governance part4_governance.py
run 10_unit_tests       -m pytest -q tests/

echo "All transcripts saved in transcripts/"
