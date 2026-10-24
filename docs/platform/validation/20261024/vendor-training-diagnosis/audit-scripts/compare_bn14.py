import json
from pathlib import Path
import sys
import torch
r=Path(__file__).parent
sys.path.insert(0,str(r/'tools'))
from compare_vendor_prefix import compare_tree
torch.set_num_threads(2)
x=torch.load(r/'bn14-reference.pt',map_location='cpu',weights_only=False)
rows=[]
for mode in ['native','reference']:
 folder=r/'first-divergence'/mode
 y=torch.load(folder/'first.pt',map_location='cpu',weights_only=False)
 row={'layout':mode,'comparisons':{key:compare_tree(x[key],y[key]) for key in x},'self_repeats_exact':json.loads((folder/'report.json').read_text())['all_repeats_exact']}
 rows.append(row)
 print(mode,row)
(r/'bn14-comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
