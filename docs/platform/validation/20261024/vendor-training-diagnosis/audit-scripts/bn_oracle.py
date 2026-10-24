import ast
import json
from pathlib import Path
import sys

import torch

root=Path(__file__).parent
sys.path.insert(0,str(root/'tools'))
from compare_vendor_prefix import compare_tensor
torch.set_num_threads(2)
payload=torch.load(root/'common/yolov5s-observed/leaf-0002.pt',map_location='cpu',weights_only=False)
expression=ast.parse(payload['repr'],mode='eval').body
module=torch.nn.BatchNorm2d(*[ast.literal_eval(n) for n in expression.args],
                            **{n.arg:ast.literal_eval(n.value) for n in expression.keywords}).double().train()
module.load_state_dict(payload['state_before'],strict=True)
x=payload['inputs'][0].double().requires_grad_(True)
output=module(x)
upstream=(torch.arange(output.numel(),dtype=torch.float32).remainder(97)/97).reshape(output.shape).double()
output.backward(upstream)
reference={name:p.grad for name,p in module.named_parameters()}
cases={}
for label,folder in [('nvidia',root/'nvidia-replay/replay/yolo-bn-native'),
                     ('metax_native',root/'replay/yolo-bn-native'),
                     ('metax_reference_layout',root/'replay/yolo-bn-reference')]:
 actual=torch.load(folder/'first.pt',map_location='cpu',weights_only=False)['parameter_gradients']
 cases[label]={name:compare_tensor(value,actual[name].double()) for name,value in reference.items()}
record={'scope':'Offline CPU FP64 BatchNorm gradient reference, not a model/runtime fallback',
        'repr':payload['repr'],'reference':{k:v.tolist() for k,v in reference.items()},'cases':cases}
(root/'bn-fp64-reference.json').write_text(json.dumps(record,indent=2)+'\n')
print(cases)
