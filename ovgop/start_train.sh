#!/usr/bin/env bash
set -euo pipefail

exec python -B main.py -c configs/disg_ovgop.py "$@"
