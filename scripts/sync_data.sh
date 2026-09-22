#!/bin/bash
# Push the BobGPT v1 dataset to the lab server. Keep partial transfers so a
# rerun can resume after a dropped connection.
set -euo pipefail

LOCAL="$(readlink -f "$(dirname "$0")/../data")/"
REMOTE=lab:/data/datasets/bobgptv1/

ssh "${REMOTE%%:*}" mkdir -p "${REMOTE#*:}"
rsync -a --partial --info=progress2 "$LOCAL" "$REMOTE"
