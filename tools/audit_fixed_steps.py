"""CPU metric and artifact audit of the explicit short 500-step experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
vendor = sys.argv[2]
stage = root/'short-500-01'
parent = json.loads((root/'one-epoch-plan.json').read_text())
binding = json.loads((root/'full-data-binding.json').read_text())
assert binding['identity']==parent['data_identity']
lanes = ['original_nvidia','new_nvidia'] if vendor=='nvidia' else ['new_metax']
output = stage/'cpu-audits'
output.mkdir(exist_ok=True)

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()

def classify(path):
    expected = {}
    for line in Path(binding['imagenet']['test']['meta']).read_text().splitlines():
        name,label = line.rsplit(' ',1)
        assert name not in expected
        expected[name] = int(label)
    seen,hits = set(),{1:0,5:0}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        assert row['filename'] not in seen and expected[row['filename']]==row['label']
        seen.add(row['filename'])
        for k in hits:
            hits[k] += int(row['label'] in row['topk_idx'][:k])
    assert len(seen)==len(expected)==50000
    return {f'top{k}':float(np.float32(value)*np.float32(100./len(seen))) for k,value in hits.items()},50000

def detect(path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(binding['coco']['test']['meta'])
    assert len(coco.imgs)==5000 and len(coco.cats)==80
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {row['image_id'] for row in rows} <= set(coco.imgs)
    evaluator = COCOeval(coco,coco.loadRes(rows),'bbox')
    evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    assert len(evaluator.params.imgIds)==5000
    names = ('AP','AP.5','AP.75','AP_S','AP_M','AP_L','AR_1','AR_10','AR','AR_S','AR_M','AR_L')
    return {'bbox.'+key:float(value) for key,value in zip(names,evaluator.stats)},5000

for lane in lanes:
    for model in ['resnet50','yolov5s','vit_b16']:
        directory = stage/'results'/lane/model
        train,infer = directory/'train/report.json',directory/'inference/inference.json'
        if not train.exists() or not infer.exists():
            continue
        destination = output/(lane+'-'+model+'.json')
        if destination.exists():
            prior = json.loads(destination.read_text())
            assert prior['report_sha256']==sha(train) and prior['inference_sha256']==sha(infer)
            continue
        report,inference = json.loads(train.read_text()),json.loads(infer.read_text())
        marker = json.loads((directory/'train/checkpoint.json').read_text())
        assert report['status']=='preflight_completed' and report['progress']['completed_steps']==500
        assert report['identity']['preflight_steps']==500
        assert report['identity']==inference['identity']==marker['identity']
        assert report['initial_state_sha256']==parent['models'][model]['initial_state_sha256']
        assert report['identity']['source_sha256']==parent['sources'][lane]['sha256']
        assert inference['weights_exact'] and inference['rank_local_buffers_restored']
        files = {name:sha(directory/'train'/name) for name in marker['files']}
        assert files==marker['files']
        assert inference['checkpoint_sha256']==files[f"checkpoint-{marker['slot']}.pt"]
        predictions = {case:sha(directory/case/'predictions/results.txt') for case in ['train','inference']}
        calculate = detect if model=='yolov5s' else classify
        metrics,count = calculate(directory/'train/predictions/results.txt')
        independent = metrics if predictions['train']==predictions['inference'] else calculate(directory/'inference/predictions/results.txt')[0]
        assert metrics==report['final_metrics'] and independent==inference['metrics']
        traces = {}
        for rank in range(2):
            path = directory/'train'/f'train.rank{rank}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [row['step'] for row in rows]==list(range(1,501))
            assert all(row['input_sha256'] for row in rows)
            traces[str(rank)] = sha(path)
        record = {'status':'audited_500_steps','model':model,'lane':lane,
                  'recorded_utc':datetime.now(timezone.utc).isoformat(),'scheduled_steps':500,
                  'optimizer_updates':report['progress']['optimizer_updates'],
                  'report_sha256':sha(train),'inference_sha256':sha(infer),'checkpoint_files':files,
                  'prediction_sha256':predictions,'predictions_identical':predictions['train']==predictions['inference'],
                  'recomputed_metrics':metrics,'inference_metrics':independent,'validation_images':count,
                  'training_trace_sha256':traces,'metrics_recompute_exact':True,
                  'model_computation':False,'full_epoch_completed':False,'convergence_claim':False}
        with destination.open('x') as stream:
            json.dump(record,stream,indent=2,sort_keys=True)
            stream.write('\n')
        print(json.dumps(record),flush=True)
