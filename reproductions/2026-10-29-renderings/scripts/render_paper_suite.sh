#!/usr/bin/env bash
# Re-run all three paper sizes and both phases in four output formats.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${PACKAGE_ROOT}/../.." && pwd)
cd "${PROJECT_ROOT}"

export PYTHON=${PYTHON:-python}
export EPISODES=${EPISODES:-5}
export GIF=${GIF:-true}
export CSV=${CSV:-true}
export PDF=${PDF:-true}
export ALGO=rmappo
export PROTECTED_R=0.5
export BEARING_SIGMA0=0.0262
export BEARING_R0=0.5
export CPF_SIGMA_A=0.2
export DPI=300
export LINEWIDTH=2.0
export MARKER_SIZE=6.0
export LEGEND_FONTSIZE=10
export LABEL_FONTSIZE=10
export LEGEND_LOC=paper
export ELLIPSE_INTERVAL=5
export ELLIPSE_SIGMA=3.0
OUTPUT_ROOT=${RENDER_ROOT:-${PACKAGE_ROOT}/replay_output}

for SIZE in 2v1 5v2 7v3; do
  DEFENDERS=${SIZE%v*}
  INTRUDERS=${SIZE#*v}
  printf 'Rendering %s Stage I: %s episodes\n' "${SIZE}" "${EPISODES}"
  DEFENDERS=${DEFENDERS} INTRUDERS=${INTRUDERS} SEED=61000 \
    MODEL_DIR="results/MPE/protected_zone_stage1/rmappo/rmappo_${SIZE}_stage1_/models" RUN_ID="" \
    OUTPUT_DIR="${OUTPUT_ROOT}/stage1" bash "${SCRIPT_DIR}/render_stage1.sh"

  case "${SIZE}" in
    2v1) STAGE2_SEED=61014 ;;
    5v2) STAGE2_SEED=61000 ;;
    7v3) STAGE2_SEED=61003 ;;
  esac
  printf 'Rendering %s Stage II: %s episodes\n' "${SIZE}" "${EPISODES}"
  DEFENDERS=${DEFENDERS} INTRUDERS=${INTRUDERS} SEED=${STAGE2_SEED} \
    MODEL_DIR="results/MPE/protected_zone_stage2/rmappo/rmappo_${SIZE}_stage2_/models" RUN_ID="" \
    OUTPUT_DIR="${OUTPUT_ROOT}/stage2" bash "${SCRIPT_DIR}/render_stage2.sh"
done

"${PYTHON}" "${SCRIPT_DIR}/summarize_render_output.py" --render-root "${OUTPUT_ROOT}" --expected-episodes "${EPISODES}"
