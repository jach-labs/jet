"""Report native diagnostic metrics without inventing an official index."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
names=['published-jet','base','trained']
data={name:json.loads((HERE/(name+'-benchmarks.json')).read_text()) for name in names}
cal={name:json.loads((HERE/name/'calibration-report.json').read_text()) for name in ['base','trained']}
lines=['# Jet 4B post-training evaluation','',
       '**Local diagnostic only; not an official Decision Index 0.2 score.**', '',
       'Frozen step-3750 adapter; 1900 identical requests across 15 datasets. '
       'Native per-benchmark metrics below use complete supported groups. '
       'All models use single-order Jet prompts and the same PyTorch readout backend.', '',
       '| Dataset | Metric | Published Jet 0.6B | Qwen3.5-4B base | Trained Jet 4B |',
       '|---|---|---:|---:|---:|']
maps={name:{b['dataset']:b for b in d['benchmarks']} for name,d in data.items()}
for dataset in sorted(maps['trained']):
    b=maps['trained'][dataset]
    values=[maps[name][dataset]['score'] for name in names]
    cells=['—' if v is None else f'{v*100:.2f}' for v in values]
    lines.append(f"| {dataset} | {b['metric']} | {' | '.join(cells)} |")
lines += ['', 'Scores are on a 0–100 display scale; compare within each row, not by averaging rows.', '',
          '| Model | Successful requests | Unsupported | Errors | Median request latency |',
          '|---|---:|---:|---:|---:|']
for name in names:
    d=data[name];c=d['counts']
    lines.append(f"| {name} | {c.get('ok',0)} / 1900 | {c.get('unsupported',0)} | {c.get('error',0)} | "
                 f"{d['successful_request_latency_ms']['median']:.1f} ms |")
lines += ['', '## Separate Jet test split', '',
          '600 test_v5_new examples, unused for this checkpoint selection or calibration. '
          'This is a local held-out test, not the Decision Index corpus.', '',
          '| Model | Accuracy | Raw NLL | Calibrated NLL |', '|---|---:|---:|---:|']
for name in ['base','trained']:
    t=cal[name]['test'];lines.append(f"| {name} | {t['accuracy']:.2%} | {t['raw_nll']:.4f} | {t['calibrated_nll']:.4f} |")
lines += ['', '## Limits and next step', '',
          'The 15-dataset diagnostic samples differ from full leaderboard denominators. MMLU, ARC-Easy and '
          'ARC-Challenge are not in the current 0.2 scored panel. Reported accuracy/F1 cannot be compared '
          'directly to entrants’ full-suite scores. The test/diagnostic label results did not select weights, '
          'temperatures, prompts, or inference order. No exact training-state matches were found in these '
          '1900 requests, but this is not proof of source-level decontamination; earlier training provenance '
          'limitations remain. Published Jet uses its published temperature; base and trained Qwen3.5 use '
          'temperatures fitted independently on calibration_v5. No shared-prefix optimization is used; '
          'latency is this local backend’s measurement, not the leaderboard’s RTX PRO 6000 measurement.', '',
          'The missing prerequisite for an official 0.2 comparison is the matching release-v2 corpus and '
          'reproducible selection/scoring package. Current Space methodology references corpus SHA256 '
          '`b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5`; the public kit documents v1. '
          'No 0.2 release branch/tag was found, and the Space publishes result/methodology files rather than '
          'the corpus. Obtain the matching recipe or authorized corpus access from the maintainers, verify '
          'its hash, then run the full suite. No model/results have been uploaded or submitted.', '',
          '[Leaderboard methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/methodology.json) · '
          '[Public reproduction kit](https://github.com/apolinario/decision-index)', '',
          'Reproduction: protocol.md, diagnostic-manifest.json, capacity-audit.json, model-specific reports, '
          'calibration files and raw logits in this directory. Full per-request outputs and environments are '
          'under artifacts/decision-index/runs/jet-4b-evaluation/.']
(HERE/'results.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
