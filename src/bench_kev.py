"""Score Jet on Kev's frozen out-of-domain suite and plot it next to the Kev family and Jev.

    jet-bench-kev --base-model models/jet --name jet
    jet-bench-kev --name qwen3-0.6b-untrained        # base model, as a baseline
    jet-plot-kev                                    # → docs/jet-vs-kev.png

The suite is transfer-v4 (development partition) from jaredpalmer/kev-suites, the one
Kev's own family chart uses: 11 sources Kev never trained on, 656 "clean" rows. Kev rows
become Jet rows with one-hot targets. Choice options without a description use the key
as the description, and a noul question's true/false criteria are appended to its
instructions (Jet's noul takes no criteria). Kev and Jev numbers come from Kev's
docs/kev-family-summary.json, copied into docs/bench/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from evaluate import adapter_base, collect_logits, metrics, print_table, softmax
from model import DEFAULT_BASE_MODEL, Jet

SUITE_REPO = "jaredpalmer/kev-suites"
SUITE_FILE = "v4/transfer-v4/development.jsonl"
RESULTS_DIR = Path("docs/bench/kev-transfer-v4")
KEV_SUMMARY = Path("docs/bench/kev-family-summary.json")

# Row order and names as in Kev's chart.
TASKS = {
    "sciq": "SciQ",
    "qnli": "QNLI",
    "contrastive_authorization": "Policy: authorization",
    "composition_held_and_or": "Rule: (A or B) and C",
    "composition_held_or_not": "Rule: (A and B) or not C",
    "tweet_offensive": "TweetEval offensive",
    "paws": "PAWS",
    "composition_held_conditional": "Rule: if A then not B else C",
    "mmlu": "MMLU, 4-way",
    "contrastive_deadline": "Policy: deadline (3-level Score)",
    "emotion": "Emotion, 6-way",
}
# Accuracy of a uniform guess, drawn as a tick on each row.
CHANCE = {t: 0.5 for t in TASKS} | {"sciq": 0.25, "mmlu": 0.25, "contrastive_deadline": 1 / 3, "emotion": 1 / 6}


def to_jet(row: dict) -> dict:
    """One Kev suite row (one question) → a Jet row with a one-hot target."""
    (q,) = row["questions"].values()
    label, criteria = q["label"], q.get("criteria")
    if q["type"] == "choice":
        keys = list(criteria)
        question = {"type": "choice", "instructions": q["instructions"],
                    "criteria": {k: v if v else k for k, v in criteria.items()}}
        target = [float(k == label) for k in keys]
    elif q["type"] == "score":
        question = {"type": "score", "instructions": q["instructions"], "criteria": criteria}
        target = [float(i == label) for i in range(len(criteria))]
    else:
        instructions = q["instructions"]
        if criteria:
            instructions += f"\nyes: {criteria['true']}\nno: {criteria['false']}"
        question = {"type": "noul", "instructions": instructions}
        target = [float(not label), float(label)]
    return {"state": row["state"], "question": question, "target": target, "source": q["src"]}


def load_suite() -> tuple[list[dict], str]:
    revision = HfApi().dataset_info(SUITE_REPO).sha
    path = hf_hub_download(SUITE_REPO, SUITE_FILE, repo_type="dataset", revision=revision)
    rows = [json.loads(line) for line in open(path)]
    return [to_jet(r) for r in rows if r["_meta"]["variant"] == "clean"], revision


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Jet on Kev's transfer-v4 suite.")
    ap.add_argument("--adapter", default=None, help="adapter dir; omit for a fused model or the base model")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--name", required=True, help="results file name, e.g. jet")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    rows, revision = load_suite()
    jet = Jet(adapter_base(args.adapter, args.base_model), args.adapter)
    logits, ms = collect_logits(jet, rows, args.batch_size)
    probs = [softmax(z, jet.temperatures[r["question"]["type"]]) for z, r in zip(logits, rows)]
    targets = [np.array(r["target"]) for r in rows]

    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[r["source"]].append(i)
    tasks = {k: metrics([probs[i] for i in ix], [targets[i] for i in ix]) for k, ix in by_task.items()}
    overall = metrics(probs, targets)
    print_table("by source", tasks)
    print_table("overall", {"all": overall})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    out.write_text(json.dumps({
        "model": args.name,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "suite": f"{SUITE_REPO}/{SUITE_FILE}@{revision}",
        "temperatures": jet.temperatures,
        "transfer_acc": overall["acc"],
        "transfer_brier": overall["brier"],
        "transfer_ece": overall["ece"],
        "ms_per_question": ms,
        "tasks": {k: tasks[k]["acc"] for k in TASKS if k in tasks},
        "n": {k: tasks[k]["n"] for k in TASKS if k in tasks},
    }, indent=2))
    print(f"wrote {out}")


def main_plot() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ap = argparse.ArgumentParser(description="Plot Jet next to the Kev family on transfer-v4.")
    ap.add_argument("--jet", default="jet", help="results name to plot as Jet")
    ap.add_argument("--baseline", default="qwen3-0.6b-untrained", help="results name of the untrained base")
    ap.add_argument("--out", type=Path, default=Path("docs/jet-vs-kev.png"))
    args = ap.parse_args()

    kev = json.loads(KEV_SUMMARY.read_text())
    jet = json.loads((RESULTS_DIR / f"{args.jet}.json").read_text())
    base_path = RESULTS_DIR / f"{args.baseline}.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None

    ink, muted, grid, track = "#1a1a1a", "#555555", "#e6e6e6", "#e9e9e9"
    jev_c, jet_c = "#F5A300", "#E0356B"
    # (label, results, color, hollow, backbone size in B params) in Kev's legend order, then Jet.
    series = [
        ("Jev", kev["jev"], jev_c, False, None),
        ("Kev-9B", kev["models"]["kev-9b"], "#0B1F4F", False, 9.7),
        ("Kev-4B", kev["models"]["kev-4b"], "#0A5CFF", False, 4.0),
        ("Kev-0.8B", kev["models"]["kev-0.8b"], "#6FA3F5", False, 0.8),
        ("Kev-8B (Qwen3)", kev["models"]["kev-8b-qwen3"], "#0B1F4F", True, 8.2),
        ("Kev-0.5B prototype", kev["models"]["kev-0.5b"], "#6FA3F5", True, 0.5),
        ("jet (0.6B)", jet, jet_c, False, 0.6),
    ]

    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "text.color": ink,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted})
    fig = plt.figure(figsize=(20, 13.4), dpi=136)
    fig.text(0.05, 0.945, "Where jet lands next to Kev and Jev, on data none of them trained on", fontsize=26, weight="medium")
    fig.text(0.05, 0.905, "Accuracy per source on Kev's frozen out-of-domain suite (transfer-v4, development partition, "
             f"{sum(jet['n'].values())} clean records). Same items for every model.", fontsize=14, color=muted)
    fig.add_artist(Line2D([0.05, 0.96], [0.88, 0.88], color="#cccccc", lw=1))

    ax = fig.add_axes([0.26, 0.17, 0.44, 0.63])
    names = list(TASKS)
    y = np.arange(len(names))[::-1]
    for yi, t in zip(y, names):
        vals = [s[1]["tasks"][t] for s in series]
        ax.plot([min(vals), max(vals)], [yi, yi], color=track, lw=5, solid_capstyle="round", zorder=1)
        ax.plot([CHANCE[t]] * 2, [yi - 0.22, yi + 0.22], color="#999999", lw=1.5, zorder=1)
        for label, res, color, hollow, _ in series:
            ax.scatter(res["tasks"][t], yi, s=150, zorder=3 if not hollow else 2,
                       facecolor="white" if hollow else color, edgecolor=color if hollow else "white", linewidth=2.2)
        jv, jt = kev["jev"]["tasks"][t], jet["tasks"][t]
        ax.text(jt, yi + 0.3, f"{jt * 100:.0f}", ha="center", va="bottom", fontsize=12, color=jet_c, weight="medium")
        ax.text(jv, yi - 0.3, f"{jv * 100:.0f}", ha="center", va="top", fontsize=12, color="#D08A00", weight="medium")
    ax.set_yticks(y, [TASKS[t] for t in names], fontsize=14, color=ink)
    ax.set_xlim(0.18, 1.04)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], fontsize=12)
    ax.grid(axis="x", color=grid, lw=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.text(0.05, 0.835, "Accuracy by source", fontsize=18, weight="medium")
    handles = [Line2D([], [], marker="o", ls="", ms=11, markerfacecolor="white" if h else c,
                      markeredgecolor=c if h else "white", markeredgewidth=2, label=lab) for lab, _, c, h, _ in series]
    handles.append(Line2D([], [], marker="|", ls="", ms=12, color="#999999", markeredgewidth=1.5, label="chance"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.17, 0.855), ncol=8, frameon=False,
               fontsize=12.5, handletextpad=0.3, columnspacing=1.2)

    # Right: overall accuracy against backbone size.
    rx = fig.add_axes([0.77, 0.36, 0.19, 0.44])
    fig.text(0.755, 0.835, "Overall vs backbone size", fontsize=18, weight="medium")
    rx.axhline(kev["jev"]["transfer_acc"], color=jev_c, lw=2)
    rx.text(10.5, kev["jev"]["transfer_acc"] + 0.008, f"Jev {kev['jev']['transfer_acc'] * 100:.1f}%",
            ha="right", va="bottom", fontsize=12, color="#D08A00")
    for label, res, color, hollow, size in series[1:]:
        rx.scatter(size, res["transfer_acc"], s=110, zorder=3, facecolor="white" if hollow else color,
                   edgecolor=color if hollow else "white", linewidth=2)
    if base:
        rx.scatter(0.6, base["transfer_acc"], s=110, zorder=3, facecolor="white", edgecolor=jet_c, linewidth=2)
        rx.annotate(f"Qwen3-0.6B untrained {base['transfer_acc'] * 100:.1f}%", (0.6, base["transfer_acc"]),
                    xytext=(10, -4), textcoords="offset points", fontsize=11, color=muted, va="top")
    rx.annotate(f"jet {jet['transfer_acc'] * 100:.1f}%", (0.6, jet["transfer_acc"]), xytext=(10, 0),
                textcoords="offset points", fontsize=12, color=jet_c, weight="medium", va="center")
    rx.set_xscale("log")
    rx.set_xlim(0.4, 12)
    rx.set_xticks([0.5, 1, 2, 4, 8], ["0.5B", "1B", "2B", "4B", "8B"], fontsize=12)
    rx.minorticks_off()
    lo = min([s[1]["transfer_acc"] for s in series[1:]] + ([base["transfer_acc"]] if base else []))
    rx.set_ylim(np.floor(lo * 10) / 10 - 0.02, 0.92)
    rx.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    rx.tick_params(length=0, labelsize=12)
    rx.grid(axis="y", color=grid, lw=1)
    rx.set_xlabel("Backbone size (log scale)", fontsize=12)
    for side in rx.spines.values():
        side.set_visible(False)

    rows = [(lab, res) for lab, res, *_ in series] + ([("Qwen3-0.6B, untrained", base)] if base else [])
    table = "\n".join(f"{lab:<22}{res['transfer_acc'] * 100:>6.1f}%{res['transfer_brier']:>8.3f}" for lab, res in rows)
    fig.text(0.755, 0.30, f"{'':<22}{'acc':>7}{'Brier':>8}\n{table}", fontsize=11.5, family="monospace",
             va="top", color=ink, linespacing=1.5)

    fig.text(0.05, 0.10, "jet is Qwen3-0.6B + LoRA, trained on jet's own public mix (train_v2); it has never seen any "
             "of these sources, policy families or rule structures.", fontsize=12.5, color=muted)
    fig.text(0.05, 0.075, "Kev: LoRA r=16 + pointer head on Qwen3.5 bases, trained on decision-v7. Kev and Jev numbers "
             "from Kev's docs/kev-family-summary.json.", fontsize=12.5, color=muted)
    fig.text(0.05, 0.05, "Suite rows converted to jet's format: option keys stand in for missing descriptions; noul "
             "true/false criteria appended to the instructions.  Regenerate: uv run jet-plot-kev", fontsize=12.5, color=muted)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
