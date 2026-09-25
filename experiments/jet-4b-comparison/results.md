# Jet 4B backbone pilot results

540 identical training rows, 270 identical validation rows, 27 equally weighted task families. BF16 LoRA rank 16, 135 updates per model, seed 240924. Complete prompts; no truncation.

| Model | Initial accuracy | Final accuracy | Initial NLL | Final NLL | Selected step |
|---|---:|---:|---:|---:|---:|
| qwen3-4b | 75.93% | 76.30% | 2.8237 | 0.6759 | 135 |
| qwen35-4b | 77.41% | 78.15% | 0.7051 | 0.6281 | 135 |

| Model | Selected median latency | Selected p95 latency | Training time | Training tok/s | Peak allocation |
|---|---:|---:|---:|---:|---:|
| qwen3-4b | 50.6 ms | 252.5 ms | 3.0 min | 948 | 9.64 GB |
| qwen35-4b | 66.4 ms | 271.7 ms | 3.4 min | 874 | 10.12 GB |

Trainable LoRA parameters: Qwen3 33,030,144; Qwen3.5 32,464,896.

Selected Qwen3.5 minus selected Qwen3 accuracy: +1.85 percentage points (paired, source-stratified bootstrap 95% interval: -1.85 to +5.56).
Selected NLL difference (Qwen3.5 minus Qwen3, negative favors Qwen3.5): -0.0479.

For this pilot, retain Qwen3.5-4B as the tentative quality-first choice: it has higher observed accuracy and lower NLL. Qwen3-4B-Instruct-2507 offers lower latency and slightly lower memory use. The accuracy and NLL difference intervals both include zero, so these data do not establish a decisive quality winner. A larger training run and independent evaluation are needed before replacing published Jet.

## Interpretation limits

These are model-selection validation measurements, not independent test results. The interval is descriptive and does not correct for checkpoint selection, correlated examples, or dataset contamination. A small single-seed pilot does not establish the winner after full training. Ten rows per source make individual source results noisy. The context cap excludes some long contract examples; this does not measure long-context quality. Probabilities are uncalibrated. The same rank adapts different numbers of parameters across architectures. Latency is synchronized batch-one forward/readout time after warmup, excluding tokenization and model loading. PyTorch peak allocation excludes the desktop and some driver/kernel allocations. Qwen3.5 uses FLA for gated delta attention and the native PyTorch causal convolution fallback; causal_conv1d is not installed. These speed measurements describe this backend, not an optimized architecture-level speed comparison. Desktop GPU activity was not stopped.

## Per-source selected accuracy

| Source | Qwen3 | Qwen3.5 |
|---|---:|---:|
| ag_news | 60% | 80% |
| banking77 | 100% | 90% |
| boolq | 70% | 90% |
| civil_comments | 100% | 100% |
| dbpedia | 100% | 100% |
| massive | 90% | 90% |
| mnli | 70% | 80% |
| stsb | 60% | 30% |
| v3:arithmetic | 70% | 80% |
| v3:boolean_rules | 100% | 90% |
| v3:code_behavior | 80% | 80% |
| v3:document_relevance | 90% | 100% |
| v3:entailment | 90% | 90% |
| v3:product_relevance | 20% | 60% |
| v3:sarcasm_irony | 50% | 40% |
| v3:stance | 40% | 50% |
| v3:tool_relevance | 100% | 100% |
| v3:tool_selection | 100% | 100% |
| v4:code_transfer | 50% | 40% |
| v4:contract_entailment | 80% | 70% |
| v4:stance_transfer | 60% | 60% |
| v4:tool_intent_routing | 90% | 90% |
| v5:adversarial_entailment | 80% | 90% |
| v5:commonsense_completion | 90% | 80% |
| v5:tool_response_preference | 80% | 90% |
| v5:wide_intent | 80% | 70% |
| yelp | 60% | 70% |

See protocol.md, models.json, data_manifest.json, and each model’s results JSON for reproducibility. Adapters and row-level probabilities are stored under adapters/jet-4b-comparison/. Published Jet weights and serving configuration were not changed.
