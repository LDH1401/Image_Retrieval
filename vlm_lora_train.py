# pylint: disable=missing-function-docstring
"""LoRA fine-tuning of the VLM reranker on FashionIQ train (Hướng 3, mục 6.2).

The VLM learns the very score vlm_rerank.py uses, s = logit("Yes") - logit("No")
of the first answer token, with the same prompt, chat template and image size
(it reuses vlm_rerank.Scorer). Each train query of vlm_lora_prepare.py gives one
group of pairs: its target (label 1), --n_neg candidates drawn from ranks
[--skip, top_k) of the zero-shot MagicLens list, re-drawn every epoch, and the
query image itself (--ref_negative; label 0 for both). The first ranks are
skipped because FashionIQ labels a single target per query, while other images
near the top often fit the text too.

LoRA (peft) sits on the attention projections q, k, v, o of the language model.
The vision tower, embeddings and head stay frozen, so each image's vision
features are computed once and kept in RAM, as vlm_rerank.py does.

Loss: binary cross-entropy on s, each positive weighted by the number of
negatives of its group (--loss bce); or a softmax over each group (--loss listwise).

Checkpoint choice: before training and every --eval_every optimizer steps, the
model scores --eval_queries dev queries x top --eval_top of the Hướng 1
candidates (the bundle.zip of vlm_rerank_ft/), keeping the dev queries whose
target is in that top. The mean AUC of the target against the other candidates
is logged, and the adapter is saved to <out_dir>/ckpt-<step>. At the end,
<out_dir>/best.json names the checkpoint with the best dev AUC, to pass to
vlm_rerank.py --adapter. Only dev queries are scored here; test is scored once,
by vlm_rerank.py, with the chosen checkpoint.

An interrupted run (Colab disconnect) resumes from the last checkpoint: the
order of the queries and the negatives depend only on --seed and the epoch, and
the random state is saved with each checkpoint.

Usage (Colab, a GPU with bfloat16 such as L4 or A100):
  python vlm_lora_train.py --train_dir /content/lora_train --val_dir /content/bundle \
    --out_dir /content/drive/MyDrive/magiclens_vlm_ft/lora_2b
"""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
import os
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers

from vlm_rerank import PROMPT, Scorer

def forward_scores(scorer, prepared):
  """Yes - No log-odds of a prepared batch; gradients reach the LoRA weights only."""
  inputs, keys = prepared
  if scorer.cache is not None:
    with torch.no_grad():  # frozen vision tower: features computed once, kept on CPU
      inputs = scorer._inputs_embeds(inputs, keys)  # pylint: disable=protected-access
  else:
    inputs = inputs.to(scorer.device)
  scorer.model(**inputs, use_cache=False)
  return scorer.last @ scorer.w.to(scorer.last.device) + scorer.b


def auc_of(scores, pos):
  """AUC of the candidate at index pos against the others (ties count half)."""
  s = np.asarray(scores)
  wins = (s < s[pos]).sum() + 0.5 * ((s == s[pos]).sum() - 1)
  return float(wins / (len(s) - 1))


def save_atomic(obj, path):
  tmp = path + ".tmp"
  torch.save(obj, tmp)
  os.replace(tmp, path)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--train_dir", required=True,
                      help="Unzipped lora_train_bundle.zip: candidates_train.json and images/.")
  parser.add_argument("--val_dir", required=True,
                      help="Unzipped bundle.zip of Hướng 1 (vlm_rerank_ft): candidates.json, images/.")
  parser.add_argument("--out_dir", required=True, help="Checkpoints and logs (put it on Drive).")
  parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
  parser.add_argument("--max_pixels", type=int, default=448 * 448,
                      help="Image size cap; keep the one vlm_rerank.py scores with.")
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
  parser.add_argument("--no_vision_cache", action="store_true",
                      help="Re-encode the images at every step (slower; for low-RAM runtimes).")
  parser.add_argument("--n_neg", type=int, default=4, help="Hard negatives per query and epoch.")
  parser.add_argument("--skip", type=int, default=5,
                      help="Top ranks never used as negatives (likely false negatives).")
  parser.add_argument("--no_ref_negative", action="store_true",
                      help="Do not add the query image itself as a negative.")
  parser.add_argument("--loss", default="bce", choices=["bce", "listwise"])
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--queries_per_step", type=int, default=5,
                      help="Query groups per optimizer step (5 x 6 pairs = 30 pairs).")
  parser.add_argument("--warmup_ratio", type=float, default=0.05)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--max_grad_norm", type=float, default=1.0)
  parser.add_argument("--lora_r", type=int, default=16)
  parser.add_argument("--lora_alpha", type=int, default=32)
  parser.add_argument("--lora_dropout", type=float, default=0.05)
  parser.add_argument("--grad_ckpt", action="store_true",
                      help="Gradient checkpointing: less GPU memory, about 30%% slower.")
  parser.add_argument("--eval_every", type=int, default=0,
                      help="Optimizer steps between dev checks (0 = twice per epoch).")
  parser.add_argument("--eval_queries", type=int, default=200)
  parser.add_argument("--eval_top", type=int, default=20)
  parser.add_argument("--eval_batch", type=int, default=10, help="Pairs per forward when scoring.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--limit", type=int, default=0, help="Only the first N train queries (tests).")
  parser.add_argument("--allow_cpu", action="store_true", help="Run without a GPU (tests only).")
  args = parser.parse_args()

  if not torch.cuda.is_available() and not args.allow_cpu:
    raise SystemExit("No GPU found. On Colab: Runtime > Change runtime type > L4 or A100.")
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cuda" and args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
    raise SystemExit(f"{torch.cuda.get_device_name()} has no bfloat16 (e.g. a T4): use an L4 "
                     "or A100 runtime. float16 is not offered: it overflows when training.")
  dtype = getattr(torch, args.dtype)

  with open(os.path.join(args.train_dir, "candidates_train.json")) as f:
    train = json.load(f)
  with open(os.path.join(args.val_dir, "candidates.json")) as f:
    val = json.load(f)
  if train["subtask"] != val["subtask"]:
    raise SystemExit(f"Train data is {train['subtask']}, dev candidates are {val['subtask']}.")
  queries = train["queries"][:args.limit or None]
  if args.skip + args.n_neg + 2 > train["top_k"]:
    raise SystemExit(f"--skip {args.skip} + --n_neg {args.n_neg} do not fit in the top-"
                     f"{train['top_k']} of candidates_train.json.")
  dev = [q for q in val["queries"] if q["split"] == "dev"
         and q["target"] in q["top"][:args.eval_top]][:args.eval_queries]
  if not dev:
    raise SystemExit(f"No dev query of {args.val_dir} has its target in the top-{args.eval_top}.")
  steps_per_epoch = math.ceil(len(queries) / args.queries_per_step)
  total = steps_per_epoch * args.epochs
  eval_every = args.eval_every or math.ceil(steps_per_epoch / 2)
  warmup = round(args.warmup_ratio * total)

  # Everything a resumed run must share with the run it continues.
  config = {k: getattr(args, k) for k in (
      "model", "max_pixels", "dtype", "n_neg", "skip", "loss", "lr", "epochs", "queries_per_step",
      "warmup_ratio", "weight_decay", "max_grad_norm", "lora_r", "lora_alpha", "lora_dropout",
      "eval_queries", "eval_top", "seed", "limit")} | {
      "vision_cache": not args.no_vision_cache, "ref_negative": not args.no_ref_negative,
      "eval_every": eval_every, "prompt": PROMPT, "transformers": transformers.__version__,
      "train": {c: train[c] for c in ("subtask", "top_k", "n_index", "ranker")}
               | {"n_queries": len(queries)},
      "dev": {"ranker": val.get("ranker"), "n_queries": len(dev)}}
  os.makedirs(args.out_dir, exist_ok=True)
  config_path = os.path.join(args.out_dir, "config.json")
  if os.path.exists(config_path):
    with open(config_path) as f:
      old = json.load(f)
    if old != config:
      changed = [k for k in config if old.get(k) != config[k]]
      raise SystemExit(f"{args.out_dir} holds a run with other settings ({', '.join(changed)}); "
                       "use the same settings to resume, or another --out_dir.")
  else:
    with open(config_path, "w") as f:
      json.dump(config, f, indent=1)

  @lru_cache(maxsize=512)
  def load(key):
    """Image of a key "t:<id>" (train bundle) or "v:<id>" (dev bundle)."""
    split, iid = key.split(":", 1)
    folder = args.train_dir if split == "t" else args.val_dir
    return Image.open(os.path.join(folder, "images", iid + ".png")).convert("RGB")

  scorer = Scorer(args.model, dtype, args.max_pixels, device, load,
                  vision_cache=not args.no_vision_cache)
  from peft import LoraConfig, get_peft_model  # pylint: disable=import-outside-toplevel
  torch.manual_seed(args.seed)  # LoRA's initial A matrices, then the dropout masks
  peft_model = get_peft_model(scorer.model, LoraConfig(
      r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
  lora_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
  if not lora_names or any("visual" in n for n in lora_names):
    raise SystemExit("LoRA did not land on the language model only; check --model.")
  if args.grad_ckpt:
    scorer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  params = [p for p in peft_model.parameters() if p.requires_grad]
  optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
  n_train = sum(p.numel() for p in params)
  print(f"{len(queries)} train queries x {1 + args.n_neg + (not args.no_ref_negative)} pairs, "
        f"{args.epochs} epochs = {total} steps of {args.queries_per_step} queries; LoRA "
        f"{n_train / 1e6:.1f}M parameters; dev check every {eval_every} steps on {len(dev)} "
        f"queries x top-{args.eval_top}; {args.dtype} on {device}"
        + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))

  def group(q, epoch):
    """Pairs and labels of one train query; the negatives depend on (seed, epoch, query)."""
    rng = random.Random(f"{args.seed}-{epoch}-{q['q']}")
    pool = [c for c in q["top"][args.skip:] if c not in (q["target"], q["candidate"])]
    negatives = rng.sample(pool, args.n_neg) + ([] if args.no_ref_negative else [q["candidate"]])
    prompt = PROMPT.format(cat=train["category"], text=q["text"])
    pairs = [(f"t:{q['candidate']}", f"t:{c}", prompt) for c in [q["target"]] + negatives]
    return pairs, [1.0] + [0.0] * len(negatives)

  def order(epoch):
    idx = list(range(len(queries)))
    random.Random(f"{args.seed}-order-{epoch}").shuffle(idx)
    return idx

  def evaluate():
    """Mean dev AUC (and target MRR) of the VLM score over the top --eval_top."""
    scorer.model.eval()
    aucs, rrs, t0 = [], [], time.time()
    jobs = []
    for q in dev:
      prompt = PROMPT.format(cat=val["category"], text=q["text"])
      cands = q["top"][:args.eval_top]
      jobs += [(q, [(f"v:{q['candidate']}", f"v:{c}", prompt)
                    for c in cands[i:i + args.eval_batch]])
               for i in range(0, len(cands), args.eval_batch)]
    scores = []
    with torch.no_grad(), ThreadPoolExecutor(max_workers=1) as pool:
      pending = pool.submit(scorer.prepare, jobs[0][1])
      for j, (q, triples) in enumerate(tqdm(jobs, desc="dev", leave=False)):
        prepared = pending.result()
        if j + 1 < len(jobs):
          pending = pool.submit(scorer.prepare, jobs[j + 1][1])
        scores += forward_scores(scorer, prepared).tolist()
        if len(scores) == args.eval_top:
          pos = q["top"].index(q["target"])
          aucs.append(auc_of(scores, pos))
          rrs.append(1 / (1 + sum(s > scores[pos] for s in scores)))
          scores = []
    scorer.model.train()
    scorer.inner.visual.eval()  # frozen, and without dropout anyway
    return {"dev_auc": float(np.mean(aucs)), "dev_mrr": float(np.mean(rrs)),
            "eval_s": round(time.time() - t0, 1)}

  log_path = os.path.join(args.out_dir, "train_log.jsonl")
  state_path = os.path.join(args.out_dir, "state.pt")

  def checkpoint(step, loss):
    """Dev check, adapter, log line, then the resume state (in that order)."""
    res = evaluate()
    name = f"ckpt-{step:05d}"
    if step:
      peft_model.save_pretrained(os.path.join(args.out_dir, name))
    line = {"step": step, "epoch": round(step / steps_per_epoch, 2), "ckpt": name if step else None,
            "train_loss": loss, **res, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(log_path, "a") as f:
      f.write(json.dumps(line) + "\n")
    save_atomic({"step": step, "lora": {n: p.detach().cpu() for n, p in peft_model.named_parameters()
                                       if p.requires_grad},
                 "optim": optim.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []},
                state_path)
    print(f"step {step}/{total}: dev AUC {res['dev_auc']:.4f}, MRR {res['dev_mrr']:.4f}"
          + ("" if loss is None else f", train loss {loss:.4f}"), flush=True)

  step = 0
  if os.path.exists(state_path):
    state = torch.load(state_path, map_location="cpu")
    step = state["step"]
    current = dict(peft_model.named_parameters())
    with torch.no_grad():
      for n, v in state["lora"].items():
        current[n].copy_(v.to(current[n].device, current[n].dtype))
    optim.load_state_dict(state["optim"])
    torch.set_rng_state(state["torch_rng"])
    if device == "cuda" and state["cuda_rng"]:
      torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(f"Resuming at step {step}/{total}.")
  scorer.model.train()
  scorer.inner.visual.eval()
  if step == 0:
    checkpoint(0, None)  # the untrained model: the baseline the checkpoints must beat

  lr_at = lambda s: args.lr * (min(1.0, (s + 1) / max(1, warmup)) if s < warmup
                               else max(0.0, (total - s) / max(1, total - warmup)))
  batches = []
  for s in range(step, total):
    epoch, offset = divmod(s, steps_per_epoch)
    ids = order(epoch)[offset * args.queries_per_step:(offset + 1) * args.queries_per_step]
    batches.append((s, [group(queries[i], epoch) for i in ids]))
  flat = [(s, g) for s, groups in batches for g in groups]
  n_groups = Counter(s for s, _ in flat)  # the last step of an epoch can be smaller
  losses, since_ckpt, ema = [], [], None  # losses: groups of the step; since_ckpt: steps
  with ThreadPoolExecutor(max_workers=1) as pool, tqdm(total=total, initial=step, unit="step") as bar:
    pending = pool.submit(scorer.prepare, flat[0][1][0]) if flat else None
    for j, (s, (triples, labels)) in enumerate(flat):
      prepared = pending.result()
      if j + 1 < len(flat):
        pending = pool.submit(scorer.prepare, flat[j + 1][1][0])
      sc = forward_scores(scorer, prepared)
      y = torch.tensor(labels, device=sc.device)
      if args.loss == "bce":
        loss = F.binary_cross_entropy_with_logits(
            sc, y, pos_weight=torch.tensor(float(len(labels) - 1), device=sc.device))
      else:
        loss = F.cross_entropy(sc[None], torch.zeros(1, dtype=torch.long, device=sc.device))
      (loss / n_groups[s]).backward()
      losses.append(loss.item())
      if j + 1 < len(flat) and flat[j + 1][0] == s:
        continue  # more groups of this step
      for g in optim.param_groups:
        g["lr"] = lr_at(s)
      torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
      optim.step()
      optim.zero_grad(set_to_none=True)
      step_loss = float(np.mean(losses))
      since_ckpt.append(step_loss)
      ema = step_loss if ema is None else 0.98 * ema + 0.02 * step_loss
      losses = []
      bar.update(1)
      bar.set_postfix(loss=f"{ema:.4f}", lr=f"{lr_at(s):.2e}")
      if (s + 1) % eval_every == 0 or s + 1 == total:
        checkpoint(s + 1, float(np.mean(since_ckpt)))  # mean train loss since the last check
        since_ckpt = []

  # The best checkpoint on dev (a step logged twice after a resume counts once, the last time).
  with open(log_path) as f:
    log = {r["step"]: r for r in map(json.loads, f)}
  trained = [r for r in log.values() if r["ckpt"]]
  if not trained:
    raise SystemExit("No checkpoint yet.")
  best = max(trained, key=lambda r: r["dev_auc"])
  summary = {"best": best["ckpt"], "dev_auc": best["dev_auc"], "dev_mrr": best["dev_mrr"],
             "untrained_dev_auc": log[0]["dev_auc"] if 0 in log else None,
             "checkpoints": [{k: r[k] for k in ("step", "ckpt", "dev_auc", "dev_mrr", "train_loss")}
                             for r in sorted(log.values(), key=lambda r: r["step"])]}
  with open(os.path.join(args.out_dir, "best.json"), "w") as f:
    json.dump(summary, f, indent=1)
  untrained = summary["untrained_dev_auc"]
  print(f"Best on dev: {best['ckpt']} (AUC {best['dev_auc']:.4f}"
        + ("" if untrained is None else f"; untrained {untrained:.4f}")
        + f"). Score it with vlm_rerank.py --adapter {os.path.join(args.out_dir, best['ckpt'])}")
