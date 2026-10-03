Conversational batch specs for the `chat` source. Staged and run like tutor_qa:
copy these two files into a staging dir next to symlinks to ../tutor_qa/{run_batch.py,validate.py,run_queue.sh},
then `MAX_TOKENS=400 ./run_queue.sh 3`. Valid batches go to data/sources/chat/.
