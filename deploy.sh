#!/bin/bash
# Deploy the bobgpt API to lab. Run from this repo on the desktop:
#   ./deploy.sh
# Deploys what is on origin/$BRANCH_NAME (push first). Host paths come from
# lab's /etc/bobgpt/host.env, written by the lab_server Ansible repo.
set -uo pipefail

BRANCH_NAME="develop"
SSH_TARGET="lab"

# Deploying origin, not this checkout: warn if local commits haven't been pushed.
git fetch -q origin "$BRANCH_NAME"
AHEAD=$(git rev-list --count "origin/${BRANCH_NAME}..${BRANCH_NAME}" 2>/dev/null || echo 0)
if [ "$AHEAD" -gt 0 ]; then
    echo "Warning: local ${BRANCH_NAME} is ${AHEAD} commit(s) ahead of origin; those won't be deployed."
fi

echo "Deploying origin/${BRANCH_NAME} ($(git log --oneline -1 "origin/${BRANCH_NAME}")) to ${SSH_TARGET}"

# The checkout on lab is deploy-only, so reset --hard (as beekeeper's deploy does).
# The venv is kept between deploys: rebuilding it would re-download torch every time.
# checkpoints/ is gitignored; on lab it links to the shared checkpoint tree.
ssh "$SSH_TARGET" "BRANCH_NAME=${BRANCH_NAME} bash -s" <<'EOF'
set -euo pipefail
source /etc/bobgpt/host.env
cd "$BOBGPT_SRC"
git fetch origin
git checkout "$BRANCH_NAME"
git reset --hard "origin/$BRANCH_NAME"
[ -x "$BOBGPT_VENV/bin/python" ] || "$BOBGPT_PYTHON" -m venv "$BOBGPT_VENV"
"$BOBGPT_VENV/bin/pip" install -q -r requirements-serve.txt
# ln -sfn into an existing real directory would put the link inside it.
[ -L checkpoints ] || [ ! -e checkpoints ] || { echo "checkpoints/ exists and is not a symlink" >&2; exit 1; }
ln -sfn "$BOBGPT_CHECKPOINT_ROOT" checkpoints
sudo systemctl restart bobgpt
EOF
SSH_RC=$?

if [ $SSH_RC -ne 0 ]; then
    echo "Deploy failed (exit ${SSH_RC}). bobgpt may be left on the old code or stopped." >&2
    echo "Check with: ssh ${SSH_TARGET} sudo systemctl status bobgpt" >&2
    exit $SSH_RC
fi

# systemctl restart returning 0 doesn't mean the model loaded, so wait for the
# API to answer. It binds where the desktop can't reach, so check from lab.
echo -n "Waiting for bobgpt to answer"
for _ in $(seq 1 30); do
    if ssh "$SSH_TARGET" 'source /etc/bobgpt/host.env && curl -sf -o /dev/null "http://$BOBGPT_HOST:$BOBGPT_PORT/v1/models"' 2>/dev/null; then
        echo " ok"
        echo "Deployed ${BRANCH_NAME}."
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo
echo "Deploy ran but bobgpt is not answering." >&2
echo "Check with: ssh ${SSH_TARGET} sudo journalctl -u bobgpt -n 50" >&2
exit 1
