# pylint: disable=missing-function-docstring
"""Hướng 3: stronger VLM rerankers on the same candidates.

The candidates stay those of the best first stage, the fine-tuned MagicLens of
Hướng 1 (vlm_rerank_ft/); only the VLM changes. Each VLM run is a scores file
of vlm_rerank.py in that folder, evaluated by vlm_rerank_eval.py:

  vlm_scores.jsonl          Qwen2-VL-2B, float16 (Hướng 1, configuration (e))
  vlm_scores_7b.jsonl       Qwen2-VL-7B, bfloat16, zero-shot (mục 6.1)
  vlm_scores_2b_lora.jsonl  Qwen2-VL-2B + LoRA trained on FashionIQ train (mục 6.2)

For each run found, its ranks are recomputed from the scores and checked
against its vlm_rerank_eval.py results (huong1_tong_hop.load), with the k and w
chosen on dev there. The runs are compared on the test half with paired
bootstraps, against the first stage (c), against each other and, when its
candidates are there, against the zero-shot MagicLens (a). Then, on the test
half: from which first-stage ranks the targets reach the top-10, what becomes
of the query image (it is in the index, as in the original evaluation), and
how alike the scores of the VLMs are.
Also reported, when present:
  vlm_scores_2b_bf16_check.jsonl  the 2B model in bfloat16 on the first queries,
                                  against its float16 scores: the scores, then
                                  the metrics they give on those queries
                                  (mục 6.1: the dtype alone should barely move
                                  them);
  vlm_lora/best.json              the dev check of each LoRA checkpoint, and
  vlm_lora/train_log.jsonl        the training time.
Runs not done yet are listed as missing.

Usage (from the magiclens/ folder, after vlm_rerank_eval.py on each run):
  python huong3_tong_hop.py
"""
from argparse import ArgumentParser
from datetime import datetime
import json
import os

import fiq_experiments as fx
from huong1_tong_hop import load
import numpy as np
import vlm_rerank_eval as ve

RUNS = {"vlm_scores.jsonl": "2B (Hướng 1)", "vlm_scores_7b.jsonl": "7B",
        "vlm_scores_2b_lora.jsonl": "2B + LoRA"}
BUCKETS = ((0, 10), (10, 20), (20, 30), (30, 50))  # 0-based first-stage ranks


def label(meta):
  name = meta["model"].split("/")[-1].replace("-Instruct", "")
  return (f"{name} {ve.SHORT_DTYPE.get(meta['dtype'], meta['dtype'])}"
          + (" + LoRA" if "adapter" in meta else ""))


def read_scores(path):
  out = {}
  with open(path) as f:
    for line in f:
      try:
        r = json.loads(line)
        out[r["q"]] = np.array(r["scores"])
      except (json.JSONDecodeError, KeyError):
        pass  # a line cut by a disconnect
  return out


def slots(data):
  """Slots of the target and of the query image in each top list (-1: not in it)."""
  slot = lambda q, key: q["top"].index(q[key]) if q[key] in q["top"][:data["top_k"]] else -1
  return (np.array([slot(q, "target") for q in data["queries"]]),
          np.array([slot(q, "candidate") for q in data["queries"]]))


def dtype_check(ref_path, check_path):
  """float16 vs bfloat16 scores of the same model on the queries of check_path."""
  ref, chk = read_scores(ref_path), read_scores(check_path)
  qs = sorted(set(ref) & set(chk))
  if not qs:
    return None
  a, b = np.stack([ref[q] for q in qs]), np.stack([chk[q][:ref[q].size] for q in qs])
  rank = lambda x: np.argsort(np.argsort(-x, axis=1), axis=1)
  spearman = [np.corrcoef(ra, rb)[0, 1] for ra, rb in zip(rank(a), rank(b))]
  return {"n_queries": len(qs), "n_pairs": int(a.size),
          "pearson": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
          "mean_abs_diff": float(np.abs(a - b).mean()), "max_abs_diff": float(np.abs(a - b).max()),
          "score_std": float(a.std()), "mean_spearman": float(np.mean(spearman)),
          "same_top1": float(np.mean(a.argmax(1) == b.argmax(1)))}


def dtype_metrics(data, ml, pos, base, check_path, runs):
  """What the evaluation sees of the dtype change, on the queries of check_path.

  The bfloat16 scores are fused with the k and w chosen for the float16 run
  (vlm_scores.jsonl); every run is shown on the same queries for scale.
  """
  ref = runs.get("vlm_scores.jsonl")
  chk = read_scores(check_path)
  sel = np.array([q["q"] in chk for q in data["queries"]])
  if ref is None or not sel.any():
    return None
  m, p, b = ml[sel], pos[sel], base[sel]
  everyone = np.ones(sel.sum(), bool)

  def one(vlm, k, w):
    r = ve.rerank_ranks(m, vlm, p, b, k, w)
    return {"k": k, "w": w, "auc_vlm": ve.auc(vlm[:, :k], p, everyone),
            "auc_fused": ve.auc(ve.fused(m, vlm, k, w), p, everyone),
            "R@1": float(np.mean(r < 1) * 100), "R@10": float(np.mean(r < 10) * 100), "ranks": r}

  fin = lambda r: (r["res"]["final"]["k"], r["res"]["final"]["w"])
  try:
    with open(check_path + ".meta.json") as f:
      chk_label = label(json.load(f))
  except FileNotFoundError:
    chk_label = ref["label"] + " (bản kiểm tra)"
  bf16 = np.stack([chk[q["q"]][:ref["vlm"].shape[1]] for q, s in zip(data["queries"], sel) if s])
  rows = {ref["label"]: one(ref["vlm"][sel], *fin(ref)), chk_label: one(bf16, *fin(ref))}
  rows.update({r["label"]: one(r["vlm"][sel], *fin(r)) for n, r in runs.items() if n != "vlm_scores.jsonl"})
  before, after = rows[ref["label"]]["ranks"] < 10, rows[chk_label]["ranks"] < 10
  split = np.array([q["split"] for q in data["queries"]])[sel]
  return {"n_queries": int(sel.sum()), "n_dev": int((split == "dev").sum()),
          "n_test": int((split == "test").sum()), "first_r10": float(np.mean(b < 10) * 100),
          "ref": ref["label"], "check": chk_label,
          "rows": {n: {c: v for c, v in x.items() if c != "ranks"} for n, x in rows.items()},
          "into_top10": int((~before & after).sum()), "out_of_top10": int((before & ~after).sum()),
          "r10_bootstrap": fx.paired_bootstrap(before, after)}


def train_minutes(log_path):
  """Minutes from the first dev check to the last checkpoint; None if the run was resumed."""
  with open(log_path) as f:
    log = [json.loads(line) for line in f if line.strip()]
  steps = [e["step"] for e in log]
  if not log or steps != sorted(set(steps)):
    return None, log
  t = lambda e: datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
  return ((t(log[-1]) - t(log[0])).total_seconds() + log[0]["eval_s"]) / 60, log


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--cand_dir", default="./vlm_rerank_ft",
                      help="candidates.json of the first stage and the VLM scores files.")
  parser.add_argument("--eval_dirs", nargs="+", default=["./output/huong1", "./output/huong3"],
                      help="Where vlm_rerank_eval.py wrote the results of the runs.")
  parser.add_argument("--zs_cand", default="./vlm_rerank/candidates.json",
                      help="candidates.json of the zero-shot MagicLens, for the comparison with (a).")
  parser.add_argument("--bf16_check", default="vlm_scores_2b_bf16_check.jsonl")
  parser.add_argument("--lora_log", default="./vlm_lora/best.json")
  parser.add_argument("--output", default="./output/huong3")
  args = parser.parse_args()

  cand_path = os.path.join(args.cand_dir, "candidates.json")
  with open(cand_path) as f:
    data = json.load(f)
  runs, missing = {}, []
  for name, short in RUNS.items():
    scores_path = os.path.join(args.cand_dir, name)
    if not os.path.exists(scores_path + ".meta.json"):
      missing.append(f"{short} (`{name}`)")
      continue
    with open(scores_path + ".meta.json") as f:
      meta = json.load(f)
    evals = [ve.output_stem(data, meta, d) + ".json" for d in args.eval_dirs]
    found = [p for p in evals if os.path.exists(p)]
    if not found:
      raise SystemExit(f"{scores_path} has no vlm_rerank_eval.py results yet: run "
                       f"vlm_rerank_eval.py --candidates {cand_path} --scores {scores_path} "
                       "--output ./output/huong3")
    _, res, ranks, split, n_in = load(cand_path, scores_path, found[0])
    scores = read_scores(scores_path)
    runs[name] = {"short": short, "label": label(meta), "meta": meta, "res": res, "ranks": ranks,
                  "eval": found[0], "n_in_top": n_in,
                  "vlm": np.stack([scores[q["q"]][:meta["top_k"]] for q in data["queries"]])}
  if not runs:
    raise SystemExit(f"No VLM scores in {args.cand_dir}.")
  firsts = [r["ranks"]["first"] for r in runs.values()]
  if any(not np.array_equal(firsts[0], f) for f in firsts[1:]):
    raise SystemExit("The runs do not share the same first-stage ranks.")
  first = firsts[0]
  dev, test = split == "dev", split == "test"
  ev = lambda r: {"dev": fx.recalls(r[dev]), "test": fx.recalls(r[test])}
  K = data["top_k"]
  ml = np.array([q["ml_scores"][:K] for q in data["queries"]])
  pos, ref = slots(data)

  zero_shot = None  # first-stage ranks of the zero-shot MagicLens, configuration (a)
  if args.zs_cand and os.path.exists(args.zs_cand):
    with open(args.zs_cand) as f:
      zs = json.load(f)
    same = lambda d: [(q["q"], q["candidate"], q["target"], q["split"]) for q in d["queries"]]
    if same(zs) != same(data):
      raise SystemExit(f"{args.zs_cand} does not hold the same queries as {cand_path}.")
    zero_shot = np.array([q["base_rank"] for q in zs["queries"]])

  rows = [("(a) MagicLens zero-shot", ev(zero_shot))] if zero_shot is not None else []
  rows.append(("(c) MagicLens tinh chỉnh (bước 1)", ev(first)))
  for r in runs.values():
    fin, only = r["res"]["final"], r["res"]["vlm_only"]
    rows += [(f"{r['label']}, chỉ VLM (top-{only['k']})", ev(r["ranks"]["vlm_only"])),
             (f"MagicLens + {r['label']} (top-{fin['k']}, w={fin['w']:g})", ev(r["ranks"]["fused"]))]

  comps = []
  for r in runs.values():
    comps += [(f"MagicLens + {r['label']} − (c)", first, r["ranks"]["fused"]),
              (f"{r['label']} chỉ VLM − (c)", first, r["ranks"]["vlm_only"])]
  names = list(runs)
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      comps.append((f"MagicLens + {runs[b]['label']} − MagicLens + {runs[a]['label']}",
                    runs[a]["ranks"]["fused"], runs[b]["ranks"]["fused"]))
  if zero_shot is not None:
    comps += [(f"MagicLens + {r['label']} − (a)", zero_shot, r["ranks"]["fused"])
              for r in runs.values()]
  boot = {c: {f"R@{n}": fx.paired_bootstrap(a[test] < n, b[test] < n) for n in (1, 10)}
          for c, a, b in comps}
  ci = lambda v: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"

  md = ["# Hướng 3: reranker mạnh hơn trên cùng tập ứng viên (FashionIQ "
        f"{data['subtask']}, MagicLens-{data['model_size']})", "",
        f"Bước 1: `{data.get('ranker')}` (top-{K}, kho {data['n_index']} ảnh, "
        f"dev {dev.sum()} / test {test.sum()} truy vấn). Chỉ VLM thay đổi. k và w của mỗi VLM "
        "được vlm_rerank_eval.py chọn trên dev; mọi số dưới đây là trên test, trừ cột dev R_avg.", ""]
  if missing:
    md += ["**Chưa có:** " + ", ".join(missing) + ".", ""]
  md += [fx.md_table(rows), "",
         "## So sánh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%)", "",
         "| So sánh | ΔR@1 | ΔR@10 |", "|---|---|---|"]
  md += [f"| {c} | {ci(boot[c]['R@1'])} | {ci(boot[c]['R@10'])} |" for c, _, _ in comps]
  md += ["", "## Từng VLM trên test", "",
         "AUC: khả năng tách ảnh đích khỏi các ứng viên khác trong top-k đã chọn (0.5 là ngẫu nhiên), "
         "trung bình trên các truy vấn có ảnh đích trong top-k. Thời gian: giờ GPU đo trên Colab, "
         "không gồm tải và nạp mô hình; mỗi lần chạy có thể dùng GPU khác nhau.", "",
         "| VLM | k | Truy vấn có ảnh đích trong top-k | AUC MagicLens | AUC VLM | AUC trộn | "
         "Tương quan | Giờ GPU | ms/cặp |", "|---|---|---|---|---|---|---|---|---|"]
  fmt = lambda v: "—" if v is None else f"{v:.3f}"
  for r in runs.values():
    a, res = r["res"]["auc_test"], r["res"]
    pairs = res["n_queries"] * r["meta"]["top_k"]
    md.append(f"| {r['label']} | {res['final']['k']} | {r['n_in_top']} | {fmt(a['MagicLens'])} | "
              f"{fmt(a['VLM'])} | {fmt(a['MagicLens + VLM'])} | {res['corr_ml_vlm_test']:.2f} | "
              f"{res['vlm_gpu_s'] / 3600:.1f} | {res['vlm_gpu_s'] / pairs * 1000:.0f} |")

  # Where the targets in the top-10 after the rerank came from.
  b = first[test]
  buckets = []
  md += ["", "## Ảnh đích lọt top-10 sau rerank, theo hạng của nó ở bước 1 (test)", "",
         "Mỗi ô: tỉ lệ truy vấn của nhóm có ảnh đích trong top-10 sau khi trộn (nhóm 1–10: giữ lại "
         "được; các nhóm sau: đưa lên được).", "",
         "| Hạng ở bước 1 | Số truy vấn | " + " | ".join(f"MagicLens + {r['label']}" for r in runs.values())
         + " |", "|---|---|" + "---|" * len(runs)]
  for lo, hi in BUCKETS:
    sel = (b >= lo) & (b < hi)
    share = {r["label"]: float(np.mean(r["ranks"]["fused"][test][sel] < 10) * 100) if sel.any() else None
             for r in runs.values()}
    buckets.append({"ranks": [lo + 1, hi], "n": int(sel.sum()), "top10": share})
    md.append(f"| {lo + 1}–{hi} | {sel.sum()} | "
              + " | ".join("—" if v is None else f"{v:.1f}%" for v in share.values()) + " |")
  md += ["", f"{int((b >= K).sum())} truy vấn test có ảnh đích ngoài top-{K} của bước 1: không "
         "cấu hình rerank nào đưa được ảnh đích của chúng vào top-10."]

  # The query image: in the index (as in the original evaluation), never the answer.
  ref_first = np.where(ref >= 0, ref, K)  # its rank in the first stage (K: below the top-K)
  qimg = [("(c) MagicLens tinh chỉnh (bước 1)", ref_first, first,
           ve.drop_query_image(first, ml, pos, ref))]
  for r in runs.values():
    k, w = r["res"]["final"]["k"], r["res"]["final"]["w"]
    qimg.append((f"MagicLens + {r['label']}", ve.rerank_ranks(ml, r["vlm"], ref, ref_first, k, w),
                 r["ranks"]["fused"],
                 ve.drop_query_image(r["ranks"]["fused"], ve.fused(ml, r["vlm"], k, w), pos, ref)))
  pct = lambda x, n: float(np.mean(x[test] < n) * 100)
  query_image = {n: {"in_top10": int((q[test] < 10).sum()), "at_rank1": int((q[test] == 0).sum()),
                     "R@1": pct(t, 1), "R@1_without": pct(f, 1),
                     "R@10": pct(t, 10), "R@10_without": pct(f, 10)} for n, q, t, f in qimg}
  md += ["", "## Ảnh truy vấn (test)", "",
         "Kho ảnh chứa cả ảnh truy vấn, như cách đánh giá gốc, dù ảnh này không bao giờ là đáp án. "
         "Cột \"lọc\": bỏ ảnh truy vấn khỏi danh sách kết quả rồi tính lại.", "",
         "| Cấu hình | Ảnh truy vấn trong top-10 | Ảnh truy vấn ở hạng 1 | R@1 | R@1 khi lọc | R@10 | "
         "R@10 khi lọc |", "|---|---|---|---|---|---|---|"]
  md += [f"| {n} | {x['in_top10']} | {x['at_rank1']} | {x['R@1']:.1f} | {x['R@1_without']:.1f} | "
         f"{x['R@10']:.1f} | {x['R@10_without']:.1f} |" for n, x in query_image.items()]

  out = {"ranker": data.get("ranker"), "missing": missing, "rows": dict(rows),
         "bootstrap_test": boot, "first_stage_buckets_test": buckets, "query_image_test": query_image,
         "runs": {n: {"label": r["label"], "vlm": r["meta"], "eval": r["eval"],
                      "final": r["res"]["final"], "vlm_only": r["res"]["vlm_only"],
                      "auc_test": r["res"]["auc_test"], "corr": r["res"]["corr_ml_vlm_test"],
                      "vlm_gpu_s": r["res"]["vlm_gpu_s"]} for n, r in runs.items()}}

  if len(runs) > 1:  # how alike the VLMs score the same pairs
    zt = {r["label"]: ve.z(r["vlm"][test, :K]).ravel() for r in runs.values()}
    corr = {a: {b: float(np.corrcoef(zt[a], zt[b])[0, 1]) for b in zt} for a in zt}
    out["corr_vlm_test"] = corr
    md += ["", "## Tương quan giữa điểm của các VLM (test)", "",
           f"Pearson giữa z(VLM) của hai VLM trên mọi cặp (truy vấn, ứng viên) trong top-{K} của "
           f"{test.sum()} truy vấn test. Để so: tương quan giữa z(MagicLens) và z(VLM) ở bảng "
           "\"Từng VLM\".", "", "| | " + " | ".join(zt) + " |", "|---|" + "---|" * len(zt)]
    md += [f"| {a} | " + " | ".join(f"{corr[a][b]:.2f}" for b in zt) + " |" for a in zt]

  check_path = os.path.join(args.cand_dir, args.bf16_check)
  ref_path = os.path.join(args.cand_dir, "vlm_scores.jsonl")
  if os.path.exists(check_path) and os.path.exists(ref_path):
    chk = dtype_check(ref_path, check_path)
    met = dtype_metrics(data, ml, pos, first, check_path, runs)
    out["bf16_check"] = {"scores": chk, "metrics": met}
    if chk:
      passed = chk["pearson"] >= 0.99 and chk["mean_spearman"] >= 0.98
      md += ["", "## Kiểm tra kiểu số: Qwen2-VL-2B ở float16 và bfloat16", "",
             f"Cùng mô hình, cùng {chk['n_queries']} truy vấn đầu ({chk['n_pairs']} cặp). Điểm float16 "
             "là của lần chạy Hướng 1, điểm bfloat16 của phần A, có thể trên GPU khác; nên phép so gồm "
             "mọi khác biệt khi tính toán, không riêng kiểu số.", "",
             f"- Tương quan Pearson giữa hai bộ điểm: {chk['pearson']:.4f}.",
             f"- Lệch tuyệt đối trung bình {chk['mean_abs_diff']:.3f}, lớn nhất {chk['max_abs_diff']:.3f} "
             f"(độ lệch chuẩn của điểm: {chk['score_std']:.2f}).",
             f"- Tương quan thứ hạng Spearman trong từng truy vấn, trung bình: {chk['mean_spearman']:.4f}; "
             f"cùng ứng viên đứng đầu theo VLM ở {chk['same_top1'] * 100:.1f}% truy vấn.",
             "- Tiêu chí đặt trước khi chạy (Pearson ≥ 0.99 và Spearman trung bình ≥ 0.98): "
             + ("đạt." if passed else "không đạt.")]
    if chk and met:
      md += ["", f"Chỉ số đánh giá trên cùng {met['n_queries']} truy vấn ({met['n_dev']} dev, "
             f"{met['n_test']} test; R@10 của bước 1 trên các truy vấn này: {met['first_r10']:.1f}). "
             "Bản bfloat16 được trộn với k và w đã chọn cho bản float16. Các VLM khác dùng k và w "
             "của chúng, để so độ lớn của các chênh lệch.", "",
             "| VLM | k, w | AUC VLM | AUC trộn | R@1 trộn | R@10 trộn |", "|---|---|---|---|---|---|"]
      md += [f"| {n} | {x['k']}, {x['w']:g} | {fmt(x['auc_vlm'])} | {fmt(x['auc_fused'])} | "
             f"{x['R@1']:.1f} | {x['R@10']:.1f} |" for n, x in met["rows"].items()]
      d = lambda n: met["rows"][n]["auc_vlm"] - met["rows"][met["ref"]]["auc_vlm"]
      md += ["", f"- Từ {met['ref']} sang {met['check']}: "
             f"{met['into_top10'] + met['out_of_top10']} truy vấn đổi trạng thái (ảnh đích vào top-10: "
             f"{met['into_top10']}, ra khỏi top-10: {met['out_of_top10']}); ΔR@10 {ci(met['r10_bootstrap'])}.",
             "- Chênh AUC VLM so với " + met["ref"] + ": "
             + ", ".join(f"{n} {d(n):+.3f}" for n in met["rows"] if n != met["ref"]) + "."]
  if os.path.exists(args.lora_log):
    with open(args.lora_log) as f:
      lora = json.load(f)
    out["lora"] = lora
    md += ["", "## Huấn luyện LoRA: kiểm tra dev của từng checkpoint", "",
           "AUC và MRR của điểm VLM trên top-20 ứng viên của bước 1, ở các truy vấn dev có ảnh đích "
           "trong top-20 (vlm_lora_train.py). Bước 0 là mô hình chưa huấn luyện.", "",
           "| Bước | Checkpoint | Loss train (trung bình từ lần kiểm tra trước) | AUC dev | MRR dev |",
           "|---|---|---|---|---|"]
    md += [f"| {c['step']} | {c['ckpt'] or 'chưa huấn luyện'} | "
           f"{'—' if c['train_loss'] is None else format(c['train_loss'], '.4f')} | "
           f"{c['dev_auc']:.4f} | {c['dev_mrr']:.4f} |" for c in lora["checkpoints"]]
    md += ["", f"Checkpoint được chọn: `{lora['best']}`."]
    log_path = os.path.join(os.path.dirname(args.lora_log), "train_log.jsonl")
    if os.path.exists(log_path):
      minutes, log = train_minutes(log_path)
      out["lora_train_minutes"] = minutes
      md += ["", f"Thời gian theo `train_log.jsonl`: {minutes:.0f} phút từ lần kiểm tra dev đầu đến "
             f"checkpoint cuối, trong đó {sum(e['eval_s'] for e in log) / 60:.1f} phút cho {len(log)} "
             "lần kiểm tra dev; không gồm tải và nạp mô hình." if minutes is not None else
             "Lần huấn luyện đã bị ngắt rồi chạy tiếp, nên `train_log.jsonl` không cho biết thời gian "
             "của một lần chạy liền mạch."]

  os.makedirs(args.output, exist_ok=True)
  stem = os.path.join(args.output, "huong3_tong_hop")
  with open(stem + ".md", "w") as f:
    f.write("\n".join(md) + "\n")
  with open(stem + ".json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
  print("\n".join(md))
  print("Results written to", stem + ".md")
