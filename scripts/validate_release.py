"""Compare a merged release against the unmerged adapter's saved label logits.

    uv run python scripts/validate_release.py \
        --case data/test_v5_new.jsonl experiments/jet-4b-evaluation/trained/test_v5_new-logits.json \
        --case data/calibration_v5.jsonl experiments/jet-4b-evaluation/trained/calibration_v5-logits.json

Takes the first N rows per question type (no outcome selection), scores them with
the release's own runtime on CUDA, and writes merge-validation.json into the release
folder. Also runs the public wrapper once on every question type. Exits non-zero
if more argmax answers flip, or probabilities move further, than allowed.
Generalized from experiments/jet-4b-full-20260924/validate_merged.py (the v6 check).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from release_files import DEFAULT_RELEASE, sync_code


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    ap.add_argument('--case', nargs=2, action='append', required=True, metavar=('DATA', 'LOGITS'),
                    help='jsonl rows and the adapter logits for the same rows, in order; repeatable')
    ap.add_argument('--per-type', type=int, default=12)
    ap.add_argument('--max-flips', type=int, default=0)
    ap.add_argument('--max-probability-difference', type=float, default=0.05)
    args = ap.parse_args()

    sync_code(args.release)
    sys.path.insert(0, str(args.release.resolve()))
    from jet import Jet
    from format import Question, label_token_ids
    from inference import encode
    from runtime import label_logits

    model = Jet(args.release)
    rows, refs = [], []
    for data, logits in args.case:
        part = [json.loads(line) for line in Path(data).open()]
        ref = json.loads(Path(logits).read_text())
        if len(part) != len(ref):
            raise SystemExit(f'{data}: {len(part)} rows but {len(ref)} reference logits')
        rows += part
        refs += ref

    counts, records = Counter(), []
    with torch.no_grad():
        for i, (row, ref) in enumerate(zip(rows, refs)):
            kind = row['question']['type']
            if counts[kind] >= args.per_type:
                continue
            counts[kind] += 1
            if kind != ref['type'] or row['source'] != ref['source']:
                raise SystemExit(f'row {i} does not match its reference logits')
            q = Question.from_dict(row['question'])
            ids = encode(model.tokenizer, row['state'], q, 10**9)
            z = label_logits(model.model, {'ids': ids, 'labels': label_token_ids(model.tokenizer, q)}).double()
            a = torch.tensor(ref['logits'], device='cuda', dtype=torch.float64)
            t = model.temperatures[kind]
            p, original = (z / t).softmax(-1), (a / t).softmax(-1)
            records.append({'test_index': i, 'type': kind, 'same_argmax': int(z.argmax()) == int(a.argmax()),
                            'max_probability_difference': float((p - original).abs().max())})
        smoke = model.decide('The item arrived broken.', {
            'category': {'type': 'choice', 'instructions': 'What happened?',
                         'criteria': {'damage': 'The item was damaged.', 'billing': 'A payment problem.'}},
            'severity': {'type': 'score', 'instructions': 'Rate the severity.', 'criteria': ['none', 'moderate', 'severe']},
            'damaged': {'type': 'noul', 'instructions': 'Is the item damaged?'},
        })
        assert set(smoke['answers']) == {'category', 'severity', 'damaged'}

    report = {
        'cases': len(records),
        'selection': f'First {args.per_type} cases per type, scanning the given files in order; no outcome selection',
        'argmax_flips': sum(not r['same_argmax'] for r in records),
        'max_probability_difference': max(r['max_probability_difference'] for r in records),
        'records': records, 'wrapper_smoke': smoke, 'base_download_required': False,
    }
    (args.release / 'merge-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['records', 'wrapper_smoke']}))
    if report['argmax_flips'] > args.max_flips or report['max_probability_difference'] > args.max_probability_difference:
        raise SystemExit('merged release differs from the adapter by more than allowed')


if __name__ == '__main__':
    main()
