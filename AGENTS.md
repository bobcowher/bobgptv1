# bobgpt-v1 project memory

- Beekeeper project name: `bobgpt-v1`.
- Primary performance metric: validation loss (`val_loss`), lower is better.
- Training goal: build a very small language model from scratch for educational
  purposes.
- Training plan: first pretrain on the combined public-domain literature and
  permissively licensed Python corpus, then perform a separate supervised
  fine-tuning phase on Python-heavy question/answer data.
- Dataset name: `bobgptv1`. The local canonical dataset is
  `/data/datasets/bobgptv1`, linked into this repository as `data/`, and synced
  to `lab:/data/datasets/bobgptv1` by `scripts/sync_data.sh`.
- Use the Beekeeper MCP tools for remote training operations and refresh project
  instructions before analyzing or managing a run.
