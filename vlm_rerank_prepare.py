# pylint: disable=missing-function-docstring
"""Exports MagicLens' top-K candidates on FashionIQ for the VLM reranker.

The first stage is the MagicLens checkpoint in --model_path: zero-shot MagicLens
by default, as in the baseline of fiq_experiments.py, or a checkpoint saved by
fiq_finetune.py (upstream preprocessing, query text "{c1} and {c2}" either way).
The CLIP embeddings come from the cache of fiq_experiments.py; they are checked
against the checkpoint's backbone, which fiq_finetune.py keeps frozen.
For every val query it keeps the top-K images with their MagicLens scores, the
target's rank and the dev/test split of fiq_experiments.py, and writes:
  <out_dir>/candidates.json  used by vlm_rerank.py (Colab) and vlm_rerank_eval.py;
                             its "ranker" field names the checkpoint, so scores
                             of other candidates are never mixed in
  <out_dir>/bundle.zip       what to upload to Google Drive: candidates.json,
                             vlm_rerank.py and the images the candidates use

Usage (from the magiclens/ folder, after fiq_experiments.py has cached the val
embeddings):
  python vlm_rerank_prepare.py --subtask dress
  # Hướng 1: candidates of the fine-tuned model, in their own folder
  python vlm_rerank_prepare.py --subtask dress \
    --model_path ./models/magic_lens_clip_base_fiq_dress_ft.pkl --out_dir ./vlm_rerank_ft
"""
from argparse import ArgumentParser
import json
import os
import zipfile

from clip.simple_tokenizer import default_bpe
import fiq_experiments as fx
from inference import load_model
import jax
import numpy as np
from model import MagicLens
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_size", default="base", choices=["base", "large"])
  parser.add_argument("--model_path", default="./models/magic_lens_clip_base.pkl")
  parser.add_argument("--subtask", default="dress", choices=list(fx.CATEGORY))
  parser.add_argument("--data_dir", default="./data/fiq")
  parser.add_argument("--cache_dir", default="./cache")
  parser.add_argument("--out_dir", default="./vlm_rerank")
  parser.add_argument("--top_k", type=int, default=50, help="Candidates kept per query.")
  parser.add_argument("--seed", type=int, default=0, help="Seed of the dev/test split.")
  args = parser.parse_args()
  ranker = os.path.basename(args.model_path)
  cand_path = os.path.join(args.out_dir, "candidates.json")
  if os.path.exists(cand_path):
    with open(cand_path) as f:
      old = json.load(f)
    # Files written before the "ranker" field came from the released checkpoint.
    old_ranker = old.get("ranker", f"magic_lens_clip_{old['model_size']}.pkl")
    if old_ranker != ranker:
      raise SystemExit(f"{cand_path} holds the candidates of {old_ranker}, which its VLM scores "
                       f"belong to; write those of {ranker} to another --out_dir.")

  ids, queries = fx.load_fiq(args.data_dir, args.subtask)
  cache = os.path.join(args.cache_dir, f"fiq-{args.subtask}-{args.model_size}")
  meta_path = os.path.join(cache, "meta.json")
  meta = {"ids": ids, "queries": [[q["candidate"], q["target"]] for q in queries]}
  stale = not os.path.exists(meta_path)
  if not stale:
    with open(meta_path) as f:
      stale = json.load(f) != meta
  if stale:
    raise SystemExit("Val cache is missing or stale: run fiq_experiments.py first.")
  index_of = {iid: n for n, iid in enumerate(ids)}
  refs = np.array([index_of[q["candidate"]] for q in queries])
  targets = np.array([index_of[q["target"]] for q in queries])
  perm = np.random.default_rng(args.seed).permutation(len(queries))
  dev = set(perm[:len(perm) // 2].tolist())

  # MagicLens scores of the checkpoint, from the cached CLIP embeddings.
  img = np.load(os.path.join(cache, "img_original.npy"))
  txt = np.load(os.path.join(cache, "txt_and.npy"))
  empty = np.load(os.path.join(cache, "txt_empty.npy"))
  model, params = load_model(args.model_size, args.model_path)
  # The cache is only valid if this checkpoint has the backbone that made it:
  # re-embed a few images and the empty text, and compare.
  enc_img = jax.jit(lambda x: model.apply(params, x, method=MagicLens.encode_image))
  enc_txt = jax.jit(lambda x: model.apply(params, x, method=MagicLens.encode_text))
  tokenizer = clip_tokenizer.build_tokenizer(bpe_path=default_bpe())
  probe_img = fx.image_embeds(enc_img, ids[:4], f"{args.data_dir}/images", "original", 4)
  probe_txt = fx.text_embeds(enc_txt, tokenizer, [""], 64)
  if not (np.allclose(probe_img, img[:4], rtol=1e-3, atol=1e-4)
          and np.allclose(probe_txt, empty, rtol=1e-3, atol=1e-4)):
    raise SystemExit(f"The CLIP embeddings cached in {cache} were not made by the backbone of "
                     f"{args.model_path}: re-create the cache with this checkpoint.")
  fuse = lambda i, t: np.asarray(model.apply(
      params, i, t, method=MagicLens.fuse)["multimodal_embed_norm"])
  scores = fuse(img[refs], txt) @ fuse(img, np.repeat(empty, len(ids), axis=0)).T
  base_ranks = fx.target_ranks(scores, targets)
  top = np.argsort(-scores, axis=1)[:, :args.top_k]
  splits = np.array(["dev" if i in dev else "test" for i in range(len(queries))])
  for s in ("dev", "test"):
    r = fx.recalls(base_ranks[splits == s])
    print(f"First stage ({ranker}) {s}: R@10 {r['R@10']:.1f}, R@50 {r['R@50']:.1f}, "
          f"R_avg {r['R_avg']:.1f}; target in top-{args.top_k}: "
          f"{np.mean(base_ranks[splits == s] < args.top_k) * 100:.1f}%")

  cat = fx.CATEGORY[args.subtask]
  candidates = {
      "subtask": args.subtask, "category": cat, "model_size": args.model_size,
      "seed": args.seed, "top_k": args.top_k, "n_index": len(ids), "ranker": ranker,
      "queries": [{
          "q": i, "split": str(splits[i]),
          "candidate": q["candidate"], "target": q["target"], "captions": q["captions"],
          "text": fx.TEMPLATES["and"].format(c1=q["captions"][0], c2=q["captions"][1], cat=cat),
          "base_rank": int(base_ranks[i]),
          "top": [ids[j] for j in top[i]],
          "ml_scores": [round(float(s), 6) for s in scores[i, top[i]]],
      } for i, q in enumerate(queries)]}

  os.makedirs(args.out_dir, exist_ok=True)
  with open(cand_path, "w") as f:
    json.dump(candidates, f, ensure_ascii=False)
  needed = sorted({q["candidate"] for q in queries} | {ids[j] for j in top.ravel()})
  bundle = os.path.join(args.out_dir, "bundle.zip")
  with zipfile.ZipFile(bundle, "w") as z:
    z.write(cand_path, "candidates.json", compress_type=zipfile.ZIP_DEFLATED)
    z.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_rerank.py"),
            "vlm_rerank.py", compress_type=zipfile.ZIP_DEFLATED)
    for iid in needed:  # PNGs are already compressed
      z.write(os.path.join(args.data_dir, "images", iid + ".png"), f"images/{iid}.png")
  print(f"{len(queries)} queries x top-{args.top_k} = {len(queries) * args.top_k} pairs, "
        f"{len(needed)} images.")
  print(f"Wrote {cand_path} and {bundle} ({os.path.getsize(bundle) / 1e6:.0f} MB).")
