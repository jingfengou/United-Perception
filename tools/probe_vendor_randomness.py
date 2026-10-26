"""Standalone native Torch RNG/Dropout and duplicate-write diagnostics, no UP."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def signature(tensor):
    value=tensor.detach().cpu().contiguous()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
            'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def configure():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_device(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    configure()
    report={'cases':{},'device':torch.cuda.get_device_name(),'torch':torch.__version__,
            'seed':131,'up_imported':False,'up_platform_imported':False,
            'scope':'Same-seed standalone primitive probes, not a model accuracy result'}
    shapes={'dropout_tokens':(64,17,768),'dropout_mlp':(64,17,3072),
            'dropout_attention':(64,12,17,17),'drop_path':(64,17,768)}
    for name,shape in shapes.items():
        previous=None
        repeats=[]
        for trial in range(10):
            torch.manual_seed(131)
            torch.cuda.manual_seed_all(131)
            before=torch.cuda.get_rng_state().clone()
            value=torch.ones(shape,device='cuda',requires_grad=True)
            if name=='drop_path':
                uniform=torch.rand((shape[0],1,1),device='cuda',dtype=value.dtype)
                noise=(.9+uniform).floor()
                output=value.div(.9)*noise
            else:
                output=torch.nn.functional.dropout(value,p=.1,training=True)
            node=output.grad_fn
            saved={key:signature(getattr(node,key)) for key in dir(node)
                   if key.startswith('_saved') and isinstance(getattr(node,key),torch.Tensor)}
            mask=(output.detach()!=0)
            output.sum().backward()
            result={'output':output.detach().cpu(),'gradient':value.grad.detach().cpu(),'mask':mask.cpu(),
                    'rng_before':before.cpu(),'rng_after':torch.cuda.get_rng_state().cpu()}
            if previous is None:
                torch.save(result,args.output/(name+'.pt'))
                report['cases'][name]={'output':signature(result['output']),'gradient':signature(result['gradient']),
                    'mask':signature(result['mask']),'zero_count':int((~result['mask']).sum()),
                    'rng_before':signature(result['rng_before']),'rng_after':signature(result['rng_after']),
                    'grad_fn':type(node).__name__,'saved_tensors':saved}
                previous=result
            else:
                repeats.append({key:torch.equal(previous[key],result[key]) for key in result})
        report['cases'][name]['repeats']=repeats
    cases={}
    for name in ('indexed_write','index_put'):
        outputs=[]
        for _ in range(10):
            target=torch.zeros((2,3,4,4),device='cuda')
            indices=tuple(torch.tensor(v,device='cuda') for v in ([0,0,0,1,1],[1,1,1,2,2],[2,2,2,3,3],[3,3,3,1,1]))
            values=torch.tensor([.1,.2,.3,.4,.5],device='cuda')
            if name=='indexed_write': target[indices]=values
            else: target.index_put_(indices,values,accumulate=False)
            outputs.append(target.cpu())
        torch.save(outputs[0],args.output/(name+'.pt'))
        cases[name]={'target':signature(outputs[0]),'selected_values':[float(outputs[0][0,1,2,3]),float(outputs[0][1,2,3,1])],
                     'exact_repeats':sum(torch.equal(outputs[0],v) for v in outputs[1:])}
    report['duplicate_write_cases']=cases
    report['up_imported']='up' in sys.modules
    report['up_platform_imported']='up_platform' in sys.modules
    assert not report['up_imported'] and not report['up_platform_imported']
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'cases':list(shapes),'up_imported':False,'duplicate_writes':cases},indent=2))


if __name__=='__main__':
    main()
