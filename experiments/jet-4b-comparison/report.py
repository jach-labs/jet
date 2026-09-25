"""Summarize the paired pilot once both runs finish."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
names = ['qwen3-4b', 'qwen35-4b']
results = [json.loads((HERE / f'{name}-results.json').read_text()) for name in names]
rows = []
for result in results:
    stage = 'final' if result['selected_step'] else 'initial'
    rows.append(json.loads((Path(result['adapter_dir']) / f'{stage}.json').read_text())['rows'])
assert [(r['index'], r['source']) for r in rows[0]] == [(r['index'], r['source']) for r in rows[1]]
rng = np.random.default_rng(240924)
groups = [[i for i, r in enumerate(rows[0]) if r['source'] == source]
          for source in sorted({r['source'] for r in rows[0]})]
deltas = {key: np.array([b[key] - a[key] for a, b in zip(*rows)]) for key in ['correct', 'nll']}
bootstrap = {key: [] for key in deltas}
for _ in range(5000):
    sampled = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
    for key, values in deltas.items():
        bootstrap[key].append(float(values[sampled].mean()))
paired = {key: {'qwen35_minus_qwen3': float(values.mean()),
                'stratified_bootstrap_95_percent_interval': np.percentile(bootstrap[key], [2.5, 97.5]).tolist()}
          for key, values in deltas.items()}
(HERE / 'paired-comparison.json').write_text(json.dumps(paired, indent=2) + '\n')
lines = ['# Jet 4B backbone pilot results', '',
         '540 identical training rows, 270 identical validation rows, 27 equally weighted task families. '
         'BF16 LoRA rank 16, 135 updates per model, seed 240924. Complete prompts; no truncation.', '',
         '| Model | Initial accuracy | Final accuracy | Initial NLL | Final NLL | Selected step |',
         '|---|---:|---:|---:|---:|---:|']
for name, r in zip(names, results):
    lines.append(f"| {name} | {r['initial']['acc']:.2%} | {r['final']['acc']:.2%} | "
                 f"{r['initial']['nll']:.4f} | {r['final']['nll']:.4f} | {r['selected_step']} |")
lines += ['', '| Model | Selected median latency | Selected p95 latency | Training time | Training tok/s | Peak allocation |',
          '|---|---:|---:|---:|---:|---:|']
for name, r in zip(names, results):
    lines.append(f"| {name} | {r['selected']['latency_median_ms']:.1f} ms | "
                 f"{r['selected']['latency_p95_ms']:.1f} ms | {r['training_seconds']/60:.1f} min | "
                 f"{r['training_tokens_per_second']:.0f} | {r['peak_allocated_gb']:.2f} GB |")
counts = [json.loads((Path(r['adapter_dir']) / 'config.json').read_text())['trainable_parameters'] for r in results]
lines += ['', f'Trainable LoRA parameters: Qwen3 {counts[0]:,}; Qwen3.5 {counts[1]:,}.']
acc = paired['correct']; nll = paired['nll']
lo, hi = acc['stratified_bootstrap_95_percent_interval']
lines += ['', f"Selected Qwen3.5 minus selected Qwen3 accuracy: {acc['qwen35_minus_qwen3']*100:+.2f} percentage points "
          f"(paired, source-stratified bootstrap 95% interval: {lo*100:+.2f} to {hi*100:+.2f}).",
          f"Selected NLL difference (Qwen3.5 minus Qwen3, negative favors Qwen3.5): {nll['qwen35_minus_qwen3']:+.4f}.", '',
          'For this pilot, retain Qwen3.5-4B as the tentative quality-first choice: it has higher observed accuracy '
          'and lower NLL. Qwen3-4B-Instruct-2507 offers lower latency and slightly lower memory use. '
          'The accuracy and NLL difference intervals both include zero, so these data do not establish a decisive '
          'quality winner. A larger training run and independent evaluation are needed before replacing published Jet.', '',
          '## Interpretation limits', '',
          'These are model-selection validation measurements, not independent test results. The interval is descriptive '
          'and does not correct for checkpoint selection, correlated examples, or dataset contamination. A small single-seed '
          'pilot does not establish the winner after full training. Ten rows per source make individual source results noisy. '
          'The context cap excludes some long contract examples; this does not measure long-context quality. '
          'Probabilities are uncalibrated. The same rank adapts different numbers of parameters across architectures. '
          'Latency is synchronized batch-one forward/readout time after warmup, excluding tokenization and model loading. '
          'PyTorch peak allocation excludes the desktop and some driver/kernel allocations. '
          'Qwen3.5 uses FLA for gated delta attention and the native PyTorch causal convolution fallback; '
          'causal_conv1d is not installed. These speed measurements describe this backend, not an optimized '
          'architecture-level speed comparison. Desktop GPU activity was not stopped.', '',
          '## Per-source selected accuracy', '', '| Source | Qwen3 | Qwen3.5 |', '|---|---:|---:|']
for source in sorted(results[0]['selected']['by_source']):
    lines.append(f"| {source} | {results[0]['selected']['by_source'][source]['acc']:.0%} | "
                 f"{results[1]['selected']['by_source'][source]['acc']:.0%} |")
lines += ['', 'See protocol.md, models.json, data_manifest.json, and each model’s results JSON for reproducibility. '
          'Adapters and row-level probabilities are stored under adapters/jet-4b-comparison/. '
          'Published Jet weights and serving configuration were not changed.']
(HERE / 'results.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
