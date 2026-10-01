#!/usr/bin/env bash
# run_queue.sh [WORKERS] — run queue.txt through run_batch.py with a few workers.
# Any batch that fails validation (e.g. Codex hit its usage limit) creates STOP;
# workers finish their current batch and start nothing new. Resumable: valid
# batches are skipped.
set -u
cd "$(dirname "$0")"
WORKERS="${1:-3}"
rm -f STOP

worker() {
  while [ ! -e STOP ]; do
    slug=$(flock queue.lock sh -c 'head -1 queue.pending; sed -i 1d queue.pending')
    [ -z "$slug" ] && break
    python run_batch.py "$slug" >> run_queue.out 2>&1
    if ! python validate.py "batches/$slug.jsonl" > /dev/null 2>&1; then
      echo "STOP: $slug failed validation" >> run_queue.out
      tail -5 "logs/$slug.log" >> run_queue.out 2>/dev/null
      rm -f "batches/$slug.jsonl"
      touch STOP
    fi
  done
}

cp "${QUEUE:-queue.txt}" queue.pending
for _ in $(seq "$WORKERS"); do worker & done
wait
echo "queue finished: $(ls batches | wc -l) batch files" >> run_queue.out
