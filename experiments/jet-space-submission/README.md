---
title: Jev Decision Index
emoji: 🔬
colorFrom: yellow
colorTo: blue
sdk: static
header: mini
pinned: false
short_description: Benchmarks and news on various repros of TypeSafe's Jev
thumbnail: https://huggingface.co/spaces/multimodalart/jev-decision-index/resolve/main/og.png
---

# Jev Reproductions Tracker

Who is rebuilding TypeSafe's **Jev** (System One / RLCD) in the open?

This static Space opens on the Decision Index leaderboard; the **News** tab tracks the artifacts in one combined grid, color-coded by kind:

- **Decoding**: parallel constrained decoding on stock models (inference technique, no new weights)
- **Diffusion**: text diffusion models run in a "Jev mode"
- **Trained**: Jev-like scoring heads and fine-tunes, weights often on the Hub, promised models listed last
- **Prior art**: "this already exists" claims
- **Explainers**: architecture speculation, explainers, benchmarks and roundups

Cards sort by a **trending** score: ♥ likes on X + 5 × GitHub stars + 8 × Hub likes + views ÷ 500, plus a **recency credit** (up to 8000, halving daily) so a fresh release is not buried under older, louder ones. The credit is scaled by the item's own engagement, so a brand-new entry nobody has reacted to yet cannot climb on its date alone. Cards added within a day of the metrics snapshot get a green **new** badge. Use the category and "has" chips to filter, or switch the sort.

…plus a section on what is still **not** in the open: TypeSafe's weights, the RLCD algorithm, and any open model matching Jev's calibration claims.

Hugging Face models and Spaces are first-class artifacts here, not only GitHub repos.

## Decision Index (`index.html`)

The **Index** tab (switch at the top of every page) is the leaderboard: every open reproduction that finished the frozen suite, scored with one number, the Decision Index, plus per-category radars against Jev and a full page per model (`index.html?model=<engine>`; `?model=jev` is Jev's own report).

All numbers come from one static bundle, `data/index.json`, written by `evaluation/reproductions/build_leaderboard.py` in the `typesafe-diffusion-lab` checkout from `capability-indices.json`, `entrant-metadata.json`, each run's `benchmark-summary.json` and the Jev release report. Re-run that script and commit the JSON to refresh the page; the HTML never needs to change for a data update.

The edition is read from the data too: `suite.edition` and `suite.label` in `data/index.json` (and `edition.label` / `suite.note` in `data/methodology.json`) drive the page titles, the header, the footers and the suite description; when those fields are absent the pages fall back to Decision Index 0.1 on release-v1. `build_leaderboard.py --suite release-v2` and `build_methodology.py --suite release-v2` write `data/index-v2.json` and `data/methodology-v2.json` (Decision Index 0.2: the same suite minus a stratified subset cut of ToolRet and BRIGHT); copying them over `index.json` and `methodology.json` switches the whole site to that edition. The site now serves 0.2 this way; the 0.1 bundle is archived as `data/index-v0.1.json` and `data/methodology-v0.1.json`, which is also where a `--suite release-v1` build now writes. The static `<title>` and `og:title` strings in the HTML are the no-JavaScript fallback and should be bumped by hand when an edition ships.

the answered-rate explanations come from `evaluation/reproductions/answer_gaps.py`, which mines every run's refusal messages into one line per cause and writes `answer-gaps.json` for the build to pick up.

Editorial choices baked into the build: the six interactive environments in the frozen panel are left out of every model's index (Jev included) because no reproduction has run them. Decision Index 0.1 is a 19-benchmark panel across five areas (ChessBench folded into Knowledge & Reasoning, Language split into Language Understanding and Retrieval & Classification). Decision Index 0.2 averages 40 benchmarks in the same five equal-weight areas (the edition panel `PANEL_V02` in `evaluation/reproductions/edition.py`): every static benchmark except MMLU, ARC-Easy, ARC-Challenge and SimpleBench, which stay on the board outside the index, plus the seven benchmarks added in 0.2; ForecastBench enters against its baseline as clip((0.25 − Brier) / 0.25) × coverage. An entrant goes on the 0.2 board only with results on all 40; iSarcasmEval's headline is track A (English sarcasm F1) with the other tracks listed under each result; 0.1 shows the balanced raw index; from 0.2 the headline is the chance-corrected index (`suite.headline` = `balanced_skill`: each benchmark mapped to (score − chance) / (1 − chance) before averaging, so 0 is random guessing and 100 perfect), with raw accuracy kept on model pages and the breadth variant computed but hidden.

## Regenerating the social image

`og-index.html` is the 1200×630 stage that produces `og.png`, the Space thumbnail (title lockup plus the Decision Index bar chart, read live from `data/index.json`; render at 2× with headless Chrome after regenerating the data). `og-news.html` is the older stage for the News tab and produces `og-news.png`.

## Contributing

All news data lives in one JS array (`ITEMS`) near the bottom of `news.html` (the News tab; the Index tab is `index.html`). Open a PR on this Space to add or correct an entry. Please include the announcement post on X if there is one, and the Hub / GitHub link.

Engagement metrics are a snapshot (2026-09-24) from public post metadata and will drift.

## Credits

Visual identity and outlined Huggies from [HF Huggiverse (Chunte/HFBA)](https://huggingface.co/spaces/Chunte/HFBA). Unofficial and community-maintained; not affiliated with TypeSafe AI.
