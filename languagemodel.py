import torch
import tiktoken
import numpy as np
from models import GPTModel
import math
import os
import time


# TEMP: GPT-2 weight loading helpers
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# pretrain.py writes here; posttrain.py starts from it. Under data/ so both
# Beekeeper projects see it.
PRETRAIN_CHECKPOINT = "data/checkpoints/pretrain/model.pth"
POSTTRAIN_CHECKPOINT = "data/checkpoints/posttrain/model.pth"


def code_version():
    """'<branch>@<short sha>' of the checkout this is running from, '+dirty' if edited."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    branch, sha = git("rev-parse", "--abbrev-ref", "HEAD"), git("rev-parse", "--short", "HEAD")
    if not sha:
        return "nogit"
    dirty = "+dirty" if git("status", "--porcelain", "--untracked-files=no") else ""
    return f"{branch}@{sha}{dirty}"


class LanguageModel:

    def __init__(self, gpt_config, train_loader=None, val_loader=None, checkpoint_path="checkpoints/model.pth",
                 lr=0.0006, compile=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TF32 for any matmul still in fp32 (outside autocast): ~free speed on Ampere+.
        torch.set_float32_matmul_precision("high")
        self.model = GPTModel(cfg=gpt_config)
        self.model.to(self.device)  # before the optimizer: fused AdamW needs params on the GPU
        # The loss path (training and eval) runs through train_model. torch.compile costs
        # a minute or two up front, so it's only worth it on long runs. The compiled
        # wrapper shares self.model's parameters; saving and generation keep using
        # self.model, so checkpoints have no _orig_mod. prefix and generation's
        # growing sequence lengths don't trigger recompiles.
        self.train_model = torch.compile(self.model) if compile else self.model
        self.optimizer = torch.optim.AdamW(
                         self.model.parameters(),
                         lr=lr, weight_decay=0.1,
                         fused=self.device.type == "cuda"
                        )
        self.train_loader = train_loader
        self.val_loader  = val_loader

        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.checkpoint_path = checkpoint_path


    def train(self, num_epochs, eval_freq, eval_iter, start_context, patience=2,
              warmup_steps=2000, min_lr_ratio=0.1, save_each_eval=False, max_steps=None,
              run_name=None):

        # Linear warmup, then cosine decay to min_lr_ratio * peak over the whole run.
        # The schedule needs the run length up front: num_epochs is the budget, or
        # max_steps when set (ablation runs train a fixed token budget, then stop).
        total_steps = max_steps or num_epochs * len(self.train_loader)

        def lr_factor(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)

        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        last_eval_time, last_eval_tokens = time.time(), 0
        best_val_loss, epochs_without_improvement = float("inf"), 0
        # Imported here so serving (which never trains) doesn't need tensorboard.
        from torch.utils.tensorboard import SummaryWriter
        # TensorBoard names a run after its directory, so the label says which code ran:
        # runs/<branch>@<sha>[_<run_name>]/.
        version = code_version()
        label = f"{version}_{run_name}" if run_name else version
        writer = SummaryWriter(log_dir=os.path.join("runs", label.replace("/", "-")))
        writer.add_text("run/code", f"{version} {run_name or ''}", 0)
        print(f"TensorBoard run: {label}")

        try:
            for epoch in range(num_epochs):
                self.model.train()
                epoch_loss_sum, epoch_batch_count = 0.0, 0

                for input_batch, target_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    loss = self.calc_loss_batch(input_batch, target_batch)
                    loss.backward()
                    self.optimizer.step()
                    scheduler.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1
                    # .item() waits for the GPU to finish, so call it once per step.
                    batch_loss = loss.item()
                    epoch_loss_sum += batch_loss
                    epoch_batch_count += 1
                    writer.add_scalar("loss/train_batch", batch_loss, global_step)

                    if global_step % eval_freq == 0:
                        train_loss, val_loss = self.evaluate_model(eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        writer.add_scalar("loss/train_eval", train_loss, global_step)
                        writer.add_scalar("loss/val", val_loss, global_step)
                        writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tok_per_sec = (tokens_seen - last_eval_tokens) / (time.time() - last_eval_time)
                        writer.add_scalar("throughput/tokens_per_sec", tok_per_sec, global_step)
                        print(f"Ep {epoch+1} (Step {global_step:06d}): "
                              f"Train loss {train_loss:.3f}, "
                              f"Val loss {val_loss:.3f}, "
                              f"LR {scheduler.get_last_lr()[0]:.2e}, "
                              f"{tok_per_sec:,.0f} tok/s"
                              )

                        sample = self.generate_and_print_sample(start_context)
                        writer.add_text("samples/generated_text", sample, global_step)
                        writer.flush()
                        if save_each_eval:  # long single-epoch runs: don't lose hours to a crash
                            self.save_the_model()
                        last_eval_time, last_eval_tokens = time.time(), tokens_seen

                    if global_step + 1 >= total_steps:
                        break

                epoch_train_loss = epoch_loss_sum / epoch_batch_count
                # Full val set: this number drives checkpointing and early stopping.
                self.model.eval()
                with torch.no_grad():
                    epoch_val_loss = self.calc_loss_loader(data_loader=self.val_loader)
                self.model.train()
                writer.add_scalar("loss/epoch_train", epoch_train_loss, epoch + 1)
                writer.add_scalar("loss/epoch_val", epoch_val_loss, epoch + 1)
                print(f"Ep {epoch+1} done: "
                      f"Epoch train loss {epoch_train_loss:.3f}, "
                      f"Epoch val loss {epoch_val_loss:.3f}"
                      )
                writer.flush()

                # Only keep the best weights; stop once val hasn't improved for `patience` epochs.
                if epoch_val_loss < best_val_loss:
                    best_val_loss, epochs_without_improvement = epoch_val_loss, 0
                    self.save_the_model()
                else:
                    epochs_without_improvement += 1
                    print(f"No val improvement for {epochs_without_improvement} epoch(s) "
                          f"(best {best_val_loss:.3f})")
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping after epoch {epoch+1}")
                        break
                if global_step + 1 >= total_steps:
                    print(f"Reached the step budget ({total_steps} steps, {tokens_seen:,} tokens)")
                    break
        finally:
            writer.close()

        return train_losses, val_losses, track_tokens_seen



    def save_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")


    def load_the_model(self, checkpoint_path=None):
        if checkpoint_path == None:
            checkpoint_path = self.checkpoint_path

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def generate_text_simple(self, idx,
                             max_new_tokens, context_size):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :] # grabs the last time step
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


    def generate_streaming(self, idx, max_new_tokens, context_size,
                 temperature=0.0, top_k=None, eos_id=None):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond)

            logits = logits[:, -1, :] # grabs the last time step

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                        logits < min_val,
                        torch.tensor(float('-inf')).to(logits.device),
                        logits
                        )
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if idx_next == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next


    def generate(self, idx, max_new_tokens, context_size,
                 temperature=0.0, top_k=None, eos_id=None):
        # Same sampling as generate_streaming, collected: returns the prompt plus
        # every new token, stopping early at eos_id.
        new_tokens = list(self.generate_streaming(idx, max_new_tokens, context_size,
                                                  temperature=temperature, top_k=top_k,
                                                  eos_id=eos_id))
        if new_tokens:
            idx = torch.cat([idx, *new_tokens], dim=1)
        return idx

    def text_to_token_ids(self, text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(self, token_ids, tokenizer):
        flat = token_ids.squeeze(0)
        return tokenizer.decode(flat.tolist())

    def calc_loss_batch(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        # bf16 autocast: weights stay fp32, matmuls run in bf16. No GradScaler needed
        # (unlike fp16). Loss is computed in fp32.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            logits = self.train_model(input_batch)
        loss = torch.nn.functional.cross_entropy(
                logits.float().flatten(0, 1), target_batch.flatten()
                )
        return loss

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.

        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches


    def evaluate_model(self, eval_iter):
        self.model.eval()

        with torch.no_grad():
            train_loss = self.calc_loss_loader(data_loader=self.train_loader, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(data_loader=self.val_loader, num_batches=eval_iter)
            self.model.train()
            return train_loss, val_loss

    def generate_and_print_sample(self, start_context):
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = self.text_to_token_ids(start_context, self.tokenizer).to(self.device)
        with torch.no_grad():
            token_ids = self.generate(
                    idx=encoded,
                    max_new_tokens=50, context_size=context_size,
                    temperature=0.8, top_k=40
                    )
        decoded_text = self.token_ids_to_text(token_ids, self.tokenizer)
        print(decoded_text.replace("\n", " "))
        self.model.train()
        return decoded_text


    # TEMP: load pretrained OpenAI GPT-2 weights from gpt_download.download_and_load_gpt2
    def load_gpt_weights(self, params):
        gpt = self.model

        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T)
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T)
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b)
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b)
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b)

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"])

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale,
                params["blocks"][b]["ln_1"]["g"])
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift,
                params["blocks"][b]["ln_1"]["b"])
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale,
                params["blocks"][b]["ln_2"]["g"])
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift,
                params["blocks"][b]["ln_2"]["b"])

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

        gpt.to(self.device)
        print("Loaded pretrained GPT-2 weights")
