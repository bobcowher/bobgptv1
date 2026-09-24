#!/usr/bin/env bash
#
# download_models.sh [RUN_ID] — fetch trained weights off the Beekeeper box.
#
#   ./download_models.sh          # latest run
#   ./download_models.sh 16       # a specific run
#
# Downloads every checkpoint the run produced, named checkpoints/run<ID>_<name>,
# then also copies each one over checkpoints/<name> so scripts that load the
# default path (e.g. LanguageModel.checkpoint_path) pick up the pulled run.
# The run-tagged copy is kept so you can always tell which run a checkpoint
# came from even after checkpoints/<name> has been overwritten again.
set -euo pipefail

BEEKEEPER_HOST="${BEEKEEPER_HOST:-http://lab.local:5000}"
PROJECT="bobgpt-v1"
RUN_ID="${1:-latest}"
DEST="$(dirname "$0")/checkpoints"

API="${BEEKEEPER_HOST}/api/v1/projects/${PROJECT}/runs/${RUN_ID}"

echo "Listing checkpoints for run '$RUN_ID' on $BEEKEEPER_HOST..."
LISTING="$(curl -fsSL "${API}/files/checkpoints")"

# Resolve "latest" to the real run id -- the listing reports which run it served,
# so the saved filenames say what they actually are.
RESOLVED="$(printf '%s' "$LISTING" | grep -oE '"run_id":[0-9]+' | head -1 | cut -d: -f2)"
NAMES="$(printf '%s' "$LISTING" | grep -oE '"name":"[^"]+"' | cut -d'"' -f4)"

if [ -z "$NAMES" ]; then
  echo "No checkpoints found for run '$RUN_ID'." >&2
  exit 1
fi

mkdir -p "$DEST"
for name in $NAMES; do
  out="$DEST/run${RESOLVED}_${name}"
  echo "  $name -> $out"
  curl -fsSL "${API}/files/checkpoints/${name}" -o "$out"
  cp "$out" "$DEST/${name}"
  echo "  (also updated $DEST/${name})"
done

echo
echo "Done."
ls -lh "$DEST"
