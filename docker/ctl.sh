#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.local.yml"

usage() {
  cat <<'EOF'
Usage:
  docker/ctl.sh build   # build local docker layers and the competition image
  docker/ctl.sh up      # build then start the competition container detached
  docker/ctl.sh train   # run training in the container and stream output here
  docker/ctl.sh play    # run the visualization demo in the X11 container
  docker/ctl.sh controller # run the competition controller with viewer
  docker/ctl.sh viz-up  # start the container with X11 visualization enabled
  docker/ctl.sh down    # stop and remove the competition container
  docker/ctl.sh enter   # open a shell inside the running container
  docker/ctl.sh viz-enter # open a shell in the visualization container
  docker/ctl.sh logs    # follow the latest training outputs.log if present
EOF
}

build_layers() {
  bash "${SCRIPT_DIR}/build.sh"
}

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

compose_viz() {
  docker compose -f "${COMPOSE_FILE}" -f "${SCRIPT_DIR}/compose.viz.yml" "$@"
}

latest_outputs_log() {
  find "${ROOT_DIR}/runs" -path "*/outputs.log" -type f 2>/dev/null | sort | tail -n 1
}

cmd="${1:-}"
case "${cmd}" in
  build)
    build_layers
    compose build
    ;;
  up)
    build_layers
    compose up -d
    ;;
  train)
    build_layers
    compose up -d
    compose exec aliengo-competition bash -lc 'cd /workspace/aliengo_competition && python scripts/train.py'
    ;;
  play)
    build_layers
    compose_viz up -d
    compose_viz exec aliengo-competition bash -lc 'cd /workspace/aliengo_competition && python scripts/play.py'
    ;;
  controller)
    build_layers
    compose_viz up -d
    compose_viz exec aliengo-competition bash -lc 'cd /workspace/aliengo_competition && python scripts/controller.py --task aliengo_flat --mode sim'
    ;;
  viz-up)
    build_layers
    compose_viz up -d
    ;;
  down)
    compose down
    ;;
  enter)
    compose exec aliengo-competition bash
    ;;
  viz-enter)
    compose_viz exec aliengo-competition bash
    ;;
  logs)
    log_file="$(latest_outputs_log)"
    if [[ -n "${log_file}" && -f "${log_file}" ]]; then
      tail -f "${log_file}"
    else
      compose logs -f aliengo-competition
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac
