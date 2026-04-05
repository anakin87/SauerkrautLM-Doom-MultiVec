"""
Train the DOOM MultiVec Classifier (encoder + classification head).

Uses standard CrossEntropy/KL-div loss instead of MaxSim-based KD.
This avoids the embedding collapse problem of the MaxSim approach.

Usage:
  # Quick test:
  python scripts/train_classifier.py --data data/doom-cls-500k --epochs 3 --batch-size 64 --bf16

  # Full training:
  python scripts/train_classifier.py --data data/doom-cls-500k --epochs 10 --batch-size 128 --bf16 --wandb
"""

import argparse
import os
import sys
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class DoomFrameDataset(Dataset):
    """Dataset of (tokenized_frame, action_scores) pairs.

    Supports both pre-tokenized datasets (with input_ids/attention_mask columns)
    and raw text datasets (tokenized on-the-fly).
    """

    def __init__(self, data_path, tokenizer=None, max_length=1100):
        from datasets import load_from_disk
        raw = load_from_disk(data_path)

        if 'input_ids' in raw.column_names:
            # Pre-tokenized: fast path, no tokenizer needed
            self.pretokenized = True
            self.input_ids = raw['input_ids']
            self.attention_mask = raw['attention_mask']
            self.scores = raw['scores']
            self.has_depth = 'depth_ids' in raw.column_names
            if self.has_depth:
                self.depth_ids = raw['depth_ids']
                print(f"  Using pre-tokenized data with DEPTH (fast)")
            else:
                print(f"  Using pre-tokenized data (fast, no depth)")
        else:
            # Raw text: tokenize on-the-fly (slow)
            self.pretokenized = False
            self.texts = raw['text']
            self.scores = raw['scores']
            self.tokenizer = tokenizer
            self.max_length = max_length
            print(f"  Using raw text data (tokenizing on-the-fly)")

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        scores = self.scores[idx]

        if self.pretokenized:
            item = {
                'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
                'labels': torch.tensor(scores, dtype=torch.float32),
            }
            if self.has_depth:
                item['depth_ids'] = torch.tensor(self.depth_ids[idx], dtype=torch.long)
            return item
        else:
            encoded = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': torch.tensor(scores, dtype=torch.float32),
            }


def evaluate(model, dataloader, device, action_names):
    """Evaluate classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    per_action_correct = {a: 0 for a in action_names}
    per_action_total = {a: 0 for a in action_names}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']  # (batch, 6) soft scores
            depth_ids = batch['depth_ids'].to(device) if 'depth_ids' in batch else None

            result = model(input_ids, attention_mask, depth_ids=depth_ids)
            preds = result['logits'].cpu().argmax(dim=-1)  # (batch,)

            # For each sample, the "valid" actions are those with score > 0.5
            for i in range(len(preds)):
                pred_idx = preds[i].item()
                pred_action = action_names[pred_idx]
                valid = {action_names[j] for j, s in enumerate(labels[i]) if s > 0.5}

                if pred_action in valid:
                    correct += 1

                for action in valid:
                    action_idx = action_names.index(action)
                    per_action_total[action] += 1
                    if pred_idx == action_idx:
                        per_action_correct[action] += 1

                total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n  Eval: {correct}/{total} = {acc:.1%} accuracy")
    for action in action_names:
        t = per_action_total[action]
        c = per_action_correct[action]
        if t > 0:
            print(f"    {action:15s}: {c}/{t} = {c/t:.0%}")

    model.train()
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train DOOM MultiVec Classifier")
    parser.add_argument('--model', default='models/doom-multivec-5L')
    parser.add_argument('--data', default='data/doom-cls-500k')
    parser.add_argument('--output', default='output/classifier-v1')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--logging-steps', type=int, default=50)
    parser.add_argument('--pool', default='attention',
                        choices=['attention', 'mean', 'cls', 'spatial', 'maxsim', 'multi_proto_attn', 'token_vote'])
    parser.add_argument('--protos', type=int, default=4,
                        help='Prototypes per action for maxsim pool mode')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--flash-attn', action='store_true',
                        help='Use Flash Attention 2 (requires GPU + flash-attn package)')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--max-length', type=int, default=1100)
    args = parser.parse_args()

    print("=" * 60)
    print("DOOM MultiVec — Classifier Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Wandb
    if args.wandb:
        import wandb
        wandb.init(project="doom-multivec", name=f"classifier-{args.pool}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Dataset
    print(f"\nLoading data from {args.data}...")
    full_dataset = DoomFrameDataset(args.data, tokenizer, args.max_length)
    print(f"  {len(full_dataset)} samples")

    # Train/eval split (95/5)
    eval_size = min(2000, len(full_dataset) // 20)
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, eval_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {train_size}, Eval: {eval_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Auto-detect num_actions from dataset
    sample_scores = full_dataset[0]['labels']
    num_actions = len(sample_scores)
    print(f"  Detected {num_actions} actions")

    # Model
    print(f"\nLoading model from {args.model}...")
    from doom_multivec.model.classifier import DoomMultiVecClassifier
    model = DoomMultiVecClassifier(
        args.model, pool_mode=args.pool, use_flash_attn=args.flash_attn,
        protos_per_action=args.protos, num_actions=num_actions,
    ).to(device)

    if args.flash_attn and device.type == 'cuda':
        print("  Enabling Flash Attention 2...")
        model.enable_flash_attn()
        print("  Flash Attention 2 enabled!")

    # No class weighting — v1 showed weighted KL-div causes collapse to shoot.

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = total_params - encoder_params
    print(f"  Encoder: {encoder_params:,} params")
    print(f"  Head: {head_params:,} params")
    print(f"  Total: {total_params:,} params")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps

    def get_lr(step):
        if step < warmup_steps:
            return max(0.01, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Mixed precision — bf16 does NOT need GradScaler (only fp16 does)
    use_scaler = args.bf16 and not args.flash_attn  # flash attn forces bf16 natively
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # disabled; bf16 is stable
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Training
    print(f"\n  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Pool: {args.pool}")
    print(f"  Steps: {total_steps}")
    print(f"  bf16: {args.bf16}")
    print(f"\nStarting training...")

    os.makedirs(args.output, exist_ok=True)
    action_names = DoomMultiVecClassifier.ACTION_NAMES[:num_actions]
    global_step = 0
    best_acc = 0.0
    running_loss = 0.0

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            depth_ids = batch['depth_ids'].to(device) if 'depth_ids' in batch else None

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=autocast_dtype, enabled=args.bf16):
                result = model(input_ids, attention_mask, labels=labels, depth_ids=depth_ids)
                loss = result['loss']

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / args.logging_steps
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step:5d} | ep {epoch+1} | loss {avg_loss:.4f} | lr {lr:.6f}")
                if args.wandb:
                    import wandb
                    wandb.log({'loss': avg_loss, 'lr': lr, 'epoch': epoch + batch_idx/len(train_loader)}, step=global_step)
                running_loss = 0.0

            if global_step % args.eval_steps == 0:
                acc = evaluate(model, eval_loader, device, action_names)
                if args.wandb:
                    import wandb
                    wandb.log({'eval_accuracy': acc}, step=global_step)

                if acc > best_acc:
                    best_acc = acc
                    save_path = os.path.join(args.output, 'best')
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
                    model.encoder.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"  New best: {acc:.1%} → saved to {save_path}")

    # Final eval
    print("\n" + "=" * 60)
    print("Final evaluation:")
    final_acc = evaluate(model, eval_loader, device, action_names)
    print(f"\nBest accuracy: {best_acc:.1%}")
    print(f"Final accuracy: {final_acc:.1%}")

    # Save final model
    final_path = os.path.join(args.output, 'final')
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_path, 'model.pt'))
    model.encoder.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Saved to {final_path}")


if __name__ == '__main__':
    main()
