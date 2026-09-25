import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];SITE=Path('/home/jach/dev/jach.me/src/jet')
ref=json.loads((ROOT/'experiments/jet-kev-comparison-20260925/reference-index.json').read_text())
comparison=json.loads((ROOT/'experiments/jet-kev-comparison-20260925/comparison.json').read_text())
models=[ref['jev']]+[next(m for m in ref['models'] if m['engine']==engine) for engine in ['kev-8b-raised','kev-9b-raised','kev-4b-raised','decider-2b-fp8-http','solomon-v11-bf16-encoding','semif','openvons']]
data={'release':'v6.1.0','referenceEdition':'Decision Index 0.1','referenceDate':ref['generated_utc'][:10],'referenceSource':comparison['reference_url'],'models':[{'name':m['name'],'overall':m['scores']['balanced_skill'],'results':{k:{'score':v.get('score'),'metric':v.get('metric'),'n':v.get('requests')} for k,v in m['results'].items()}} for m in models],'benchmarks':[]}
for r in comparison['comparison']:
 bid=str(r['id']);spec=ref['benchmarks'][bid]
 data['benchmarks'].append({'id':bid,'name':r['dataset'],'metric':r['local_metric'] or r['reference_metric'],'score':r['jet_score'],'n':r['requests'],'answered':r['answered'],'unsupported':r['unsupported'],'sampled':r['scope'].startswith('sampled'),'description':spec.get('explainer','')})
p=SITE/'index.html';s=p.read_text();payload=json.dumps(data,separators=(',',':'),ensure_ascii=False).replace('<','\\u003c')
s=re.sub(r'(<script type="application/json" id="jet-index-data">).*?(</script>)',lambda m:m[1]+payload+m[2],s,flags=re.S)
s=s.replace('Qwen3.5-4B with BF16 LoRA. Jet’s sampled diagnostic results are compared with published models; its official overall index has not yet been measured.', 'Jet v6.1.0 · full merged Qwen3.5-4B model. Results cover 25 local benchmarks, including two retrieval samples, alongside published model scores. Jet has no official overall index score.')
s=s.replace('Official overall index</option>','Archived overall index (0.1)</option>').replace('24 September 2026 snapshot','Jet: 25 September 2026 · references: archived 0.1').replace('Jet · sampled diagnostic','Jet · local evaluation')
s=s.replace('<div class="source text-small"><a href="https://huggingface.co/spaces/multimodalart/jev-decision-index" target="_blank" rel="noreferrer">Decision Index source</a></div>', '<p class="text-small text-muted">Scores measure the selected step-2,000 adapter before the BF16 merge. Merge validation is recorded in the model card. Matching metrics and case counts do not verify identical source cases; this is not an official ranking.</p>\n  <div class="source text-small"><a href="'+comparison['reference_url']+'" target="_blank" rel="noreferrer">Archived Decision Index source</a> · <a href="https://github.com/michaljach/jet/blob/main/experiments/jet-kev-comparison-20260925/results.md">Evaluation report</a> · <a href="https://huggingface.co/michaljach/jet">Jet v6.1.0 model card</a></div>')
s=s.replace('calibrated, typed answers back.', 'typed answers and probabilities back.')
s=s.replace('Temperature scaling on held-out data calibrates the probabilities. On the server, when several\nquestions share a state, the state is encoded once and reused for each of them.', 'The release uses inherited temperature scaling for its probabilities. Each complete question is processed separately, with an 8,192-token limit and no input truncation.')
s=s.replace('The base model is Qwen3-0.6B, with LoRA adapters trained on public classification datasets\nreframed as choice, score and yes/no questions.', 'The base model is Qwen3.5-4B. The trained LoRA updates are merged into the full Jet weights, so no separate adapter or base-model download is required.')
s=s.replace('src="benchmarks.js"','src="benchmarks.js?v=6.1.0"')
p.write_text(s)
(SITE/'benchmark-data.json').write_text(json.dumps(data,indent=2)+'\n')
