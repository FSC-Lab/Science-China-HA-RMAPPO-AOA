#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${PACKAGE_ROOT}/../.." && pwd)
cd "${PROJECT_ROOT}"

# ----------------------------
# Stage-II rendering helper
# ----------------------------

ALGO=${ALGO:-rmappo}              # rmappo / mappo / ippo
PYTHON=${PYTHON:-python}          # interpreter from the activated environment
DEFENDERS=${DEFENDERS:-7}         # swarm sizes: 2v1, 5v2, 7v3 (set DEFENDERS/INTRUDERS accordingly)
INTRUDERS=${INTRUDERS:-3}         # swarm sizes: 2v1, 5v2, 7v3 (set DEFENDERS/INTRUDERS accordingly)
PROTECTED_R=${PROTECTED_R:-0.5}   # protected-zone radius: 0.5, 1.0, 1.5, 2.0 (run separately)
BEARING_SIGMA0=${BEARING_SIGMA0:-0.0262} # paper nominal noise in radians
BEARING_R0=${BEARING_R0:-0.5}     # reference range for AoA noise scaling (m)
CPF_SIGMA_A=${CPF_SIGMA_A:-0.2}
RUN_ID=${RUN_ID:-}              # optional run identifier used during training
EPISODES=${EPISODES:-5}         # number of episodes to render
# Anchors reproduce the previously selected successful paper-style ep01.
case "${DEFENDERS}v${INTRUDERS}" in
  2v1) CASE_SEED=61014 ;;
  5v2) CASE_SEED=61000 ;;
  7v3) CASE_SEED=61003 ;;
  *) CASE_SEED=61000 ;;
esac
SEED=${SEED:-${CASE_SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-${PACKAGE_ROOT}/replay_output/stage2}
GIF=${GIF:-true}                # set to false to skip GIF export
PDF=${PDF:-true}                # set to false to skip vector PDF export
GIF_FPS=${GIF_FPS:-12}
GIF_STRIDE=${GIF_STRIDE:-2}
DPI=${DPI:-300}
LINEWIDTH=${LINEWIDTH:-2.0}
MARKER_SIZE=${MARKER_SIZE:-6.0}
LEGEND_FONTSIZE=${LEGEND_FONTSIZE:-10}
LABEL_FONTSIZE=${LABEL_FONTSIZE:-10}
LEGEND_LOC=${LEGEND_LOC:-paper}
ELLIPSE_INTERVAL=${ELLIPSE_INTERVAL:-5}
ELLIPSE_SIGMA=${ELLIPSE_SIGMA:-3.0}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_ROOT}/results/MPE}
MODEL_DIR=${MODEL_DIR:-}
CSV=${CSV:-true}                # set to false to skip CSV export

ARGS=(
  --algo ${ALGO}
  --defenders ${DEFENDERS}
  --intruders ${INTRUDERS}
  --episodes ${EPISODES}
  --seed ${SEED}
  --protected_r ${PROTECTED_R}
  --bearing_sigma0 ${BEARING_SIGMA0}
  --bearing_r0 ${BEARING_R0}
  --cpf_sigma_a ${CPF_SIGMA_A}
  --gif_fps ${GIF_FPS}
  --gif_stride ${GIF_STRIDE}
  --dpi ${DPI}
  --linewidth ${LINEWIDTH}
  --marker_size ${MARKER_SIZE}
  --legend_fontsize ${LEGEND_FONTSIZE}
  --label_fontsize ${LABEL_FONTSIZE}
  --legend_loc "${LEGEND_LOC}"
  --ellipse_interval ${ELLIPSE_INTERVAL}
  --ellipse_sigma ${ELLIPSE_SIGMA}
  --results_root "${RESULTS_ROOT}"
  --output_dir "${OUTPUT_DIR}"
)

if [ -n "${RUN_ID}" ]; then
  ARGS+=(--run_id "${RUN_ID}")
fi

if [ -n "${MODEL_DIR}" ]; then
  ARGS+=(--model_dir "${MODEL_DIR}")
fi

if [ "${GIF}" = "true" ]; then
  ARGS+=(--gif)
fi

if [ "${PDF}" = "true" ]; then
  ARGS+=(--pdf)
fi

if [ "${CSV}" = "true" ]; then
  ARGS+=(--csv)
fi

"${PYTHON}" "${SCRIPT_DIR}/render_protected_zone_stage2.py" "${ARGS[@]}"
