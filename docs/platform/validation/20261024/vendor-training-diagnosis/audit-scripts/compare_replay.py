import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).parent/'tools'))
from compare_vendor_prefix import compare_tree
r=Path(__file__).parent
rows=[]
for label in ['vit-conv','vit-dropout','yolo-conv','yolo-bn']:
 x=torch.load(r/'nvidia-replay'/'replay'/(label+'-native')/'first.pt',map_location='cpu',weights_only=False)
 for layout in ['native','reference']:
  folder=r/'replay'/(label+'-'+layout)
  y=torch.load(folder/'first.pt',map_location='cpu',weights_only=False)
  result={key:compare_tree(x[key],y[key]) for key in x}
  info=json.loads((folder/'report.json').read_text())
  row={'case':label,'layout':layout,'self_repeats_exact':info['all_repeats_exact'],'comparisons':result}
  rows.append(row)
  print(label,layout,'self-exact',row['self_repeats_exact'],{key:[(v.get('max_abs'),v['passed']) for v in data] for key,data in result.items()})
(r/'replay-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
