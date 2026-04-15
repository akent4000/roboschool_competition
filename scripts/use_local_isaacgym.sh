#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with 'source', for example:" >&2
  echo "  source scripts/use_local_isaacgym.sh /path/to/isaacgym" >&2
  exit 1
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

prepend_path_var() {
  local var_name="$1"
  local new_path="$2"
  local current_value="${!var_name:-}"

  if [[ -z "${new_path}" || ! -e "${new_path}" ]]; then
    return 0
  fi

  case ":${current_value}:" in
    *":${new_path}:"*) ;;
    *)
      if [[ -n "${current_value}" ]]; then
        printf -v "${var_name}" '%s:%s' "${new_path}" "${current_value}"
      else
        printf -v "${var_name}" '%s' "${new_path}"
      fi
      export "${var_name}"
      ;;
  esac
}

ISAACGYM_ROOT="${1:-${ISAACGYM_PATH:-}}"

if [[ -z "${ISAACGYM_ROOT}" ]]; then
  for candidate in "${HOME}/isaacgym" "/opt/isaacgym"; do
    if [[ -d "${candidate}/python/isaacgym" ]]; then
      ISAACGYM_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ISAACGYM_ROOT}" || ! -d "${ISAACGYM_ROOT}/python/isaacgym" ]]; then
  echo "Isaac Gym not found. Pass the path explicitly or set ISAACGYM_PATH." >&2
  echo "Expected directory layout: <isaacgym>/python/isaacgym" >&2
  return 1
fi

export ISAACGYM_PATH="${ISAACGYM_ROOT}"

prepend_path_var PYTHONPATH "${ISAACGYM_ROOT}/python"
prepend_path_var PYTHONPATH "${REPO_ROOT}/src"
prepend_path_var PYTHONPATH "${REPO_ROOT}"

if [[ -d "${ISAACGYM_ROOT}/python/isaacgym/_bindings" ]]; then
  while IFS= read -r binding_dir; do
    prepend_path_var LD_LIBRARY_PATH "${binding_dir}"
  done < <(find "${ISAACGYM_ROOT}/python/isaacgym/_bindings" -mindepth 1 -maxdepth 1 -type d | sort)
fi

echo "Configured local Isaac Gym environment:"
echo "  ISAACGYM_PATH=${ISAACGYM_PATH}"
echo "  PYTHONPATH updated with Isaac Gym and project paths"
echo "  LD_LIBRARY_PATH updated with Isaac Gym bindings"
