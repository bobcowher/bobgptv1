#!/bin/bash
# Updates and restarts the bobgpt API on lab. Run it on lab:
#   ssh lab /opt/bobgpt/src/deploy.sh
# Host paths come from /etc/bobgpt/host.env (written by the lab_server Ansible repo).
set -euo pipefail

# The branch lab serves. Work reaches it when it's ready for lab.
BRANCH=develop

source /etc/bobgpt/host.env
cd "$BOBGPT_SRC"

# --ff-only: if the checkout on lab has diverged, stop rather than merge.
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# Build the venv once; later deploys only install what changed.
[ -x "$BOBGPT_VENV/bin/python" ] || "$BOBGPT_PYTHON" -m venv "$BOBGPT_VENV"
"$BOBGPT_VENV/bin/pip" install -q -r requirements-serve.txt

# checkpoints/ is gitignored; on lab it links to the shared checkpoint tree.
ln -sfn "$BOBGPT_CHECKPOINT_ROOT" checkpoints

sudo systemctl restart bobgpt

# systemctl reports success even if the server then dies loading the model,
# so wait for the API to answer. The model loads at startup, so a 200 here
# means the weights loaded.
for _ in $(seq 60); do
    if curl -sf -o /dev/null "http://$BOBGPT_HOST:$BOBGPT_PORT/v1/models"; then
        echo "bobgpt is up on $BOBGPT_HOST:$BOBGPT_PORT ($(git log --oneline -1))"
        exit 0
    fi
    sleep 1
done

echo "bobgpt did not answer within 60s. Last log lines:"
sudo journalctl -u bobgpt -n 30 --no-pager
exit 1
