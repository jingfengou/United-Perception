import json
from pathlib import Path
import sys

import torch

root=Path(__file__).parent
sys.path.insert(0,str(root/'tools'))
from compare_vendor_prefix import compare_tree
torch.set_num_threads(2)
records={}
for model in ('vit_b16','yolov5s'):
 rows=[]
 for label,first,second in (
   ('nvidia_observed_control',root/(model+'-observed'),root/(model+'-control')),
   ('nvidia_original_new',root/(model+'-observed'),root/(model+'-original')),
   ('metax_observed_control',root/'metax'/(model+'-observed'),root/'metax'/(model+'-control'))):
  a=torch.load(first/'gradients-step1.pt',map_location='cpu',weights_only=True)
  b=torch.load(second/'gradients-step1.pt',map_location='cpu',weights_only=True)
  comparison=compare_tree(a,b)
  x=json.loads((first/'observations.json').read_text())['steps'][0]
  y=json.loads((second/'observations.json').read_text())['steps'][0]
  rows.append({'label':label,'first_step':{'reference':x,'actual':y},
               'gradient_tensors':len(comparison),'gradient_tensors_exact':sum(r['exact'] for r in comparison),
               'gradient_tensors_within_tolerance':sum(r['passed'] for r in comparison),
               'gradients':comparison})
 records[model]=rows
(root/'controls-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
for model,rows in records.items():
 for row in rows:print(model,row['label'],row['gradient_tensors_exact'],row['gradient_tensors_within_tolerance'],'/',row['gradient_tensors'])
