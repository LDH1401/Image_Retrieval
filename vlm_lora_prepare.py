# pylint: disable=missing-function-docstring
"""Training data for the LoRA fine-tuning of the VLM reranker (Hướng 3, mục 6.2).

For each FashionIQ train query, the zero-shot MagicLens ranks the train index
(the train split images, ~10.9K for dress) and its top-K is kept. On Colab,
vlm_lora_train.py turns each query into one positive pair (its target, "Yes")
and a few negatives ("No") drawn from that top-K, re-drawn every epoch.

The candidates come from the released checkpoint, not from the one fine-tuned
on this train split: on the data it was trained on, the fine-tuned model finds
the targets too easily, so its lists would not look like those met at test
time. Nothing of the val split is used.

Writes <out_dir>/candidates_train.json and <out_dir>/lora_train_bundle.zip
(candidates_train.json, vlm_rerank.py, vlm_lora_train.py and the images), to
upload to Google Drive next to the bundle.zip of Hướng 1.

Usage (from the magiclens/ folder, after fiq_finetune.py has cached the train
CLIP embeddings):
  python vlm_lora_prepare.py --subtask dress --out_dir ./vlm_lora
"""
from argparse import ArgumentParser
import json
import os
import zipfile

from clip.simple_tokenizer import default_bpe
import fiq_experiments as fx
from fiq_finetune import load_split
from inference import load_model
import jax
from model import MagicLens
import numpy as np
from scenic.projects.baselines.clip import tokenizer as clip_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--model_size", default="base", choices=["base", "large"])
  parser.add_argument("--model_path", default="./models/magic_lens_clip_base.pkl",
                      help="First stage that picks the negatives: the released (zero-shot) one.")
  parser.add_argument("--subtask", default="dress", choices=list(fx.CATEGORY))
  parser.add_argument("--data_dir", default="./data/fiq")
  parser.add_argument("--cache_dir", default="./cache")
  parser.add_argument("--out_dir", default="./vlm_lora")
  parser.add_argument("--top_k", type=int, default=50, help="Candidates kept per train query.")
  args = parser.parse_args()

  ids, queries, refs, targets = load_split(args.data_dir, args.subtask, "train")
  cache = os.path.join(args.cache_dir, f"fiq-{args.subtask}-{args.model_size}")
  meta_path = os.path.join(cache, "train_meta.json")
  meta = {"ids": ids, "queries": [[q["candidate"], q["target"]] for q in queries]}
  if not os.path.exists(meta_path):
    raise SystemExit(f"No train cache in {cache}: run fiq_finetune.py --subtask {args.subtask} "
                     "first (it caches the CLIP embeddings of the train split).")
  with open(meta_path) as f:
    if json.load(f) != meta:
      raise SystemExit(f"The train cache in {cache} does not match the images on disk: "
                       "delete train_*.npy and re-run fiq_finetune.py.")
  img = np.load(os.path.join(cache, "train_img_original.npy"))
  txt = np.load(os.path.join(cache, "train_txt_and.npy"))
  empty = np.load(os.path.join(cache, "txt_empty.npy"))

  model, params = load_model(args.model_size, args.model_path)
  # As in vlm_rerank_prepare.py: the cache is only valid for the backbone that made it.
  enc_img = jax.jit(lambda x: model.apply(params, x, method=MagicLens.encode_image))
  enc_txt = jax.jit(lambda x: model.apply(params, x, method=MagicLens.encode_text))
  tokenizer = clip_tokenizer.build_tokenizer(bpe_path=default_bpe())
  probe_img = fx.image_embeds(enc_img, ids[:4], f"{args.data_dir}/images", "original", 4)
  probe_txt = fx.text_embeds(enc_txt, tokenizer, [""], 64)
  if not (np.allclose(probe_img, img[:4], rtol=1e-3, atol=1e-4)
          and np.allclose(probe_txt, empty, rtol=1e-3, atol=1e-4)):
    raise SystemExit(f"The CLIP embeddings cached in {cache} were not made by the backbone of "
                     f"{args.model_path}.")
  fuse = lambda i, t: np.asarray(model.apply(
      params, i, t, method=MagicLens.fuse)["multimodal_embed_norm"])
  scores = fuse(img[refs], txt) @ fuse(img, np.repeat(empty, len(ids), axis=0)).T
  ranks = fx.target_ranks(scores, targets)
  top = np.argsort(-scores, axis=1)[:, :args.top_k]
  ref_pos = np.array([list(top[i]).index(refs[i]) if refs[i] in top[i] else -1
                      for i in range(len(queries))])
  r = fx.recalls(ranks)
  print(f"Zero-shot first stage on the train index ({len(ids)} images): R@10 {r['R@10']:.1f}, "
        f"R@50 {r['R@50']:.1f}; query image in the top-{args.top_k} of "
        f"{np.mean(ref_pos >= 0) * 100:.1f}% of the queries.")

  cat = fx.CATEGORY[args.subtask]
  candidates = {
      "subtask": args.subtask, "category": cat, "model_size": args.model_size, "split": "train",
      "top_k": args.top_k, "n_index": len(ids), "ranker": os.path.basename(args.model_path),
      "queries": [{
          "q": i, "candidate": q["candidate"], "target": q["target"], "captions": q["captions"],
          "text": fx.TEMPLATES["and"].format(c1=q["captions"][0], c2=q["captions"][1], cat=cat),
          "target_rank": int(ranks[i]),
          "top": [ids[j] for j in top[i]],
      } for i, q in enumerate(queries)]}
  os.makedirs(args.out_dir, exist_ok=True)
  cand_path = os.path.join(args.out_dir, "candidates_train.json")
  with open(cand_path, "w") as f:
    json.dump(candidates, f, ensure_ascii=False)

  needed = sorted({q["candidate"] for q in queries} | {q["target"] for q in queries}
                  | {ids[j] for j in top.ravel()})
  bundle = os.path.join(args.out_dir, "lora_train_bundle.zip")
  with zipfile.ZipFile(bundle, "w") as z:
    z.write(cand_path, "candidates_train.json", compress_type=zipfile.ZIP_DEFLATED)
    for script in ("vlm_rerank.py", "vlm_lora_train.py"):
      z.write(os.path.join(HERE, script), script, compress_type=zipfile.ZIP_DEFLATED)
    for iid in needed:  # PNGs are already compressed
      z.write(os.path.join(args.data_dir, "images", iid + ".png"), f"images/{iid}.png")
  print(f"{len(queries)} train queries, top-{args.top_k} each, {len(needed)} images.")
  print(f"Wrote {cand_path} and {bundle} ({os.path.getsize(bundle) / 1e6:.0f} MB).")
