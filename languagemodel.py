import torch
import tiktoken
from models import *

class LanguageModel:

    def __init__(self, gpt_config, train_loader, val_loader):

        torch.manual_seed(123)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPTModel(cfg=gpt_config)
        self.optimizer = torch.optim.AdamW(
                         self.model.parameters(),
                         lr=0.0004, weight_decay=0.1
                        )
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader  = val_loader

        self.tokenizer = tiktoken.get_encoding("gpt2")


    def train(self, num_epochs, eval_freq, eval_iter, start_context):

        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(num_epochs):
            self.model.train()

            for input_batch, target_batch in self.train_loader:
                self.optimizer.zero_grad()
                loss = calc_loss_batch(
                        input_batch, target_batch, self.model, self.device
                        )
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                            self.model, self.train_loader, self.val_loader, self.device, eval_iter
                            )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}"
                          )

                    generate_and_print_sample(
                            self.model, self.tokenizer, self.device, start_context
                            )

        return train_losses, val_losses, track_tokens_seen


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


    def generate(self, idx, max_new_tokens, context_size,
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

        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), target_batch.flatten()
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
        encoded = text_to_token_ids(start_context, self.tokenizer).to(self.device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                    model=self.model, idx=encoded,
                    max_new_tokens=50, context_size=context_size
                    )
        decoded_text = token_ids_to_text(token_ids, self.tokenizer)
        print(decoded_text.replace("\n", " "))
        self.model.train()

