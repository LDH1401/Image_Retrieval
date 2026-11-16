# pylint: disable=missing-function-docstring
"""Hướng 1: the five configurations (a)-(e) on the same FashionIQ test queries.

  (a) zero-shot MagicLens                     vlm_rerank/candidates.json
  (b) (a) + VLM rerank                        + vlm_rerank/vlm_scores.jsonl
  (c) fine-tuned MagicLens (fiq_finetune.py)  vlm_rerank_ft/candidates.json
  (d) (c) reordered by the VLM alone          + vlm_rerank_ft/vlm_scores.jsonl
  (e) (c) + VLM rerank

The rerank depth k and weight w of (b), (d) and (e) are read from the results
of vlm_rerank_eval.py, which chose them on dev. This script only recomputes the
ranks, checks that they give the same recalls as those results, and compares
the configurations with paired bootstraps on the test half.

Usage (from the magiclens/ folder, after vlm_rerank_eval.py has evaluated both
candidate sets):
  python huong1_tong_hop.py
"""
from argparse import ArgumentParser
import json
import os

import fiq_experiments as fx
import numpy as np
import vlm_rerank_eval as ve


def load(cand_path, scores_path, eval_path):
  """Ranks of the first stage, of the fused rerank and of the VLM alone."""
  with open(cand_path) as f:
    data = json.load(f)
  with open(eval_path) as f:
    res = json.load(f)
  with open(scores_path + ".meta.json") as f:
    meta = json.load(f)
  if res["vlm"] != meta or res.get("ranker") != data.get("ranker"):
    raise SystemExit(f"{eval_path} was not computed from {cand_path} and {scores_path}: "
                     "re-run vlm_rerank_eval.py on them.")
  if not res["complete"]:
    raise SystemExit(f"{eval_path} covers only part of the queries: finish the Colab run first.")
  scores = {}
  with open(scores_path) as f:
    for line in f:
      try:
        r = json.loads(line)
        scores[r["q"]] = r["scores"]
      except (json.JSONDecodeError, KeyError):
        pass  # a truncated last line; vlm_rerank_eval.py checked the rest
  qs, K = data["queries"], meta["top_k"]
  ml = np.array([q["ml_scores"][:K] for q in qs])
  vlm = np.array([scores[q["q"]] for q in qs])
  base = np.array([q["base_rank"] for q in qs])
  pos = np.array([q["top"].index(q["target"]) if q["target"] in q["top"][:K] else -1
                  for q in qs])
  k, w, only_k = res["final"]["k"], res["final"]["w"], res["vlm_only"]["k"]
  ranks = {"first": base, "fused": ve.rerank_ranks(ml, vlm, pos, base, k, w),
           "vlm_only": ve.rerank_ranks(ml, vlm, pos, base, only_k, ve.VLM_ONLY)}
  split = np.array([q["split"] for q in qs])
  for name, key in (("first", "baseline"), ("fused", "final"), ("vlm_only", "vlm_only")):
    for s in ("dev", "test"):
      got, want = fx.recalls(ranks[name][split == s]), res[key][s]
      if any(not np.isclose(got[c], want[c]) for c in fx.COLS):
        raise SystemExit(f"Recomputed {name} ranks disagree with {eval_path} on {s}.")
  n_in_top = int(((pos >= 0) & (pos < k))[split == "test"].sum())
  return data, res, ranks, split, n_in_top


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--zs_dir", default="./vlm_rerank",
                      help="candidates.json and VLM scores of the zero-shot first stage.")
  parser.add_argument("--ft_dir", default="./vlm_rerank_ft",
                      help="candidates.json and VLM scores of the fine-tuned first stage.")
  parser.add_argument("--scores_name", default="vlm_scores.jsonl",
                      help="File name of the VLM scores in both folders.")
  parser.add_argument("--zs_eval", default="./output/fiq-dress-base_vlm_rerank.json")
  parser.add_argument("--ft_eval", default="",
                      help="vlm_rerank_eval.py result of the fine-tuned candidates "
                           "(default: the name it writes in --output).")
  parser.add_argument("--output", default="./output/huong1")
  args = parser.parse_args()

  ft_cand = os.path.join(args.ft_dir, "candidates.json")
  if not args.ft_eval:
    with open(ft_cand) as f:
      head = json.load(f)
    args.ft_eval = os.path.join(args.output, f"fiq-{head['subtask']}-{head['model_size']}_vlm_rerank_"
                                + os.path.splitext(head["ranker"])[0] + ".json")
  zs = load(os.path.join(args.zs_dir, "candidates.json"),
            os.path.join(args.zs_dir, args.scores_name), args.zs_eval)
  ft = load(ft_cand, os.path.join(args.ft_dir, args.scores_name), args.ft_eval)
  (zs_data, zs_res, zs_ranks, split, zs_n), (ft_data, ft_res, ft_ranks, _, ft_n) = zs, ft

  # Both candidate sets must hold the same queries, split and index.
  same = lambda d: [(q["q"], q["candidate"], q["target"], q["split"]) for q in d["queries"]]
  if same(zs_data) != same(ft_data) or any(
      zs_data[c] != ft_data[c] for c in ("subtask", "model_size", "seed", "n_index")):
    raise SystemExit("The two candidate sets do not cover the same queries and index.")
  if zs_data.get("ranker") == ft_data.get("ranker"):
    raise SystemExit("Both candidate sets come from the same first stage.")
  if zs_res["vlm"]["model"] != ft_res["vlm"]["model"]:
    print("Warning: the two rerank runs used different VLMs.")

  dev, test = split == "dev", split == "test"
  ranks = {"a": zs_ranks["first"], "b": zs_ranks["fused"], "c": ft_ranks["first"],
           "d": ft_ranks["vlm_only"], "e": ft_ranks["fused"]}
  ev = lambda r: {"dev": fx.recalls(r[dev]), "test": fx.recalls(r[test])}
  name_w = lambda w: f"w={w:g}"
  zk, zw = zs_res["final"]["k"], zs_res["final"]["w"]
  fk, fw, fok = ft_res["final"]["k"], ft_res["final"]["w"], ft_res["vlm_only"]["k"]
  rows = [("(a) MagicLens zero-shot", ev(ranks["a"])),
          (f"(b) (a) + rerank VLM, top-{zk} ({name_w(zw)})", ev(ranks["b"])),
          ("(c) MagicLens tinh chỉnh", ev(ranks["c"])),
          (f"(d) (c), xếp lại chỉ theo VLM, top-{fok}", ev(ranks["d"])),
          (f"(e) (c) + rerank VLM, top-{fk} ({name_w(fw)})", ev(ranks["e"]))]

  hits = {n: {"R@10": (r[test] < 10).astype(float),
              "R_avg": ((r[test] < 10).astype(float) + (r[test] < 50)) / 2}
          for n, r in ranks.items()}
  comps = [
      ("e−c", "(e) − (c): rerank khi bước 1 đã tinh chỉnh", lambda c: (hits["c"][c], hits["e"][c])),
      ("b−a", "(b) − (a): rerank khi bước 1 là zero-shot", lambda c: (hits["a"][c], hits["b"][c])),
      ("(e−c)−(b−a)", "[(e) − (c)] − [(b) − (a)]: mức tăng nhờ rerank thay đổi bao nhiêu",
       lambda c: (hits["c"][c] + hits["b"][c] - hits["a"][c], hits["e"][c])),
      ("d−c", "(d) − (c): chỉ VLM so với bước 1 đã tinh chỉnh", lambda c: (hits["c"][c], hits["d"][c])),
      ("e−b", "(e) − (b): thêm tinh chỉnh khi đã có rerank", lambda c: (hits["b"][c], hits["e"][c])),
      ("e−a", "(e) − (a): cả hai cải tiến so với zero-shot", lambda c: (hits["a"][c], hits["e"][c])),
  ]
  boot = {key: {c: fx.paired_bootstrap(*pair(c)) for c in ("R@10", "R_avg")}
          for key, _, pair in comps}
  ci = lambda b: f"{b[0]:+.1f} [{b[1]:+.1f}, {b[2]:+.1f}]"

  def analysis_row(name, res, k, n):
    a = res["auc_test"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    return (f"| {name} | {k} | {n} | {fmt(a['MagicLens'])} | {fmt(a['VLM'])} | "
            f"{fmt(a['MagicLens + VLM'])} | {res['corr_ml_vlm_test']:.2f} |")

  vlm = ft_res["vlm"]
  md = [f"# Hướng 1: ghép tinh chỉnh với rerank bằng VLM (FashionIQ {ft_data['subtask']}, "
        f"MagicLens-{ft_data['model_size']})",
        "",
        f"Bước 1 tinh chỉnh: `{ft_data['ranker']}`. VLM: `{vlm['model']}` ({vlm['dtype']}), chấm "
        f"top-{vlm['top_k']} của mỗi truy vấn, cùng cấu hình với lần rerank zero-shot. Kho ảnh "
        f"{ft_data['n_index']} ảnh; truy vấn dev {dev.sum()} / test {test.sum()}. k và w của "
        "(b), (d), (e) được vlm_rerank_eval.py chọn trên dev; mọi con số dưới đây là trên test, "
        "trừ cột dev R_avg.",
        "", fx.md_table(rows),
        "",
        "## So sánh trên test (bootstrap ghép cặp 2000 lần, khoảng tin cậy 95%, đơn vị điểm %)",
        "",
        "| So sánh | ΔR@10 | ΔR_avg |", "|---|---|---|"]
  md += [f"| {label} | {ci(boot[key]['R@10'])} | {ci(boot[key]['R_avg'])} |"
         for key, label, _ in comps]
  md += ["",
         "Rerank chỉ xếp lại top-50 nên không đổi R@50: trần R@10 sau rerank là R@50 của bước 1 "
         f"({ev(ranks['a'])['test']['R@50']:.1f} với zero-shot, "
         f"{ev(ranks['c'])['test']['R@50']:.1f} với bản tinh chỉnh).",
         "",
         "## Hai điểm số còn bổ sung cho nhau không (test)",
         "",
         "AUC: khả năng tách ảnh đích khỏi các ứng viên khác trong top-k, trung bình trên các truy "
         "vấn có ảnh đích trong top-k (0.5 là ngẫu nhiên). Tương quan: Pearson giữa z(MagicLens) "
         "và z(VLM) trên mọi cặp (truy vấn, ứng viên) trong top-k.",
         "",
         "| Bước 1 | k | Truy vấn có ảnh đích trong top-k | AUC MagicLens | AUC VLM | "
         "AUC trộn | Tương quan |", "|---|---|---|---|---|---|---|",
         analysis_row("Zero-shot", zs_res, zk, zs_n),
         analysis_row("Tinh chỉnh", ft_res, fk, ft_n),
         "",
         f"Ảnh đích được rerank đưa vào top-10 / đẩy ra khỏi top-10: zero-shot "
         f"{zs_res['moved_test']['into_top10']} / {zs_res['moved_test']['out_of_top10']}, tinh chỉnh "
         f"{ft_res['moved_test']['into_top10']} / {ft_res['moved_test']['out_of_top10']} truy vấn."]
  md = "\n".join(md) + "\n"

  os.makedirs(args.output, exist_ok=True)
  stem = os.path.join(args.output, "huong1_tong_hop")
  with open(stem + ".md", "w") as f:
    f.write(md)
  with open(stem + ".json", "w") as f:
    json.dump({"ranker": ft_data["ranker"], "vlm": vlm,
               "configs": {k: ev(r) for k, r in ranks.items()},
               "choices": {"b": {"k": zk, "w": zw}, "d": {"k": fok}, "e": {"k": fk, "w": fw}},
               "bootstrap_test": boot,
               "auc_test": {"zero_shot": zs_res["auc_test"], "finetuned": ft_res["auc_test"]},
               "corr_ml_vlm_test": {"zero_shot": zs_res["corr_ml_vlm_test"],
                                    "finetuned": ft_res["corr_ml_vlm_test"]}},
              f, ensure_ascii=False, indent=1)
  print(md)
  print("Results written to", stem + ".md")
