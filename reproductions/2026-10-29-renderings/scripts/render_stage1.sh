#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PACKAGE_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT=$(cd "${PACKAGE_ROOT}/../.." && pwd)
cd "${PROJECT_ROOT}"

# ----------------------------
# Stage-I rendering helper
# ----------------------------

ALGO=${ALGO:-rmappo}            # rmappo / mappo / ippo
PYTHON=${PYTHON:-python}        # interpreter from the activated environment
DEFENDERS=${DEFENDERS:-5}       # swarm sizes: 2v1, 5v2, 7v3 (set DEFENDERS/INTRUDERS accordingly)
INTRUDERS=${INTRUDERS:-2}       # swarm sizes: 2v1, 5v2, 7v3 (set DEFENDERS/INTRUDERS accordingly)
PROTECTED_R=${PROTECTED_R:-0.5} # protected-zone radius: 0.5, 1.0, 1.5, 2.0 (run separately)
RUN_ID=${RUN_ID:-}              # optional run identifier used during training
EPISODES=${EPISODES:-5}         # number of episodes to render
SEED=${SEED:-61000}             # episode k uses SEED + k - 1
OUTPUT_DIR=${OUTPUT_DIR:-${PACKAGE_ROOT}/replay_output/stage1}
GIF=${GIF:-true}                # set to false to skip GIF export
PDF=${PDF:-true}                # set to false to skip vector PDF export
CSV=${CSV:-true}                # trajectories plus terminal-state records
GIF_FPS=${GIF_FPS:-12}
DPI=${DPI:-300}
LINEWIDTH=${LINEWIDTH:-2.0}
MARKER_SIZE=${MARKER_SIZE:-6.0}
LEGEND_FONTSIZE=${LEGEND_FONTSIZE:-10}
LABEL_FONTSIZE=${LABEL_FONTSIZE:-10}
LEGEND_LOC=${LEGEND_LOC:-paper}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_ROOT}/results/MPE}
MODEL_DIR=${MODEL_DIR:-}

ARGS=(
  --algo ${ALGO}
  --defenders ${DEFENDERS}
  --intruders ${INTRUDERS}
  --episodes ${EPISODES}
  --seed ${SEED}
  --protected_r ${PROTECTED_R}
  --gif_fps ${GIF_FPS}
  --dpi ${DPI}
  --linewidth ${LINEWIDTH}
  --marker_size ${MARKER_SIZE}
  --legend_fontsize ${LEGEND_FONTSIZE}
  --label_fontsize ${LABEL_FONTSIZE}
  --legend_loc "${LEGEND_LOC}"
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

"${PYTHON}" "${SCRIPT_DIR}/render_protected_zone_stage1.py" "${ARGS[@]}"
