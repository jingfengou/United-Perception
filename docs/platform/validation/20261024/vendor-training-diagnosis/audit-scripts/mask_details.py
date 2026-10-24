import json
from pathlib import Path
import torch
r=Path(__file__).parent
x=torch.load(r/'nvidia-replay/replay/vit-dropout-native/first.pt',map_location='cpu',weights_only=False)['mask'].reshape(-1)
y=torch.load(r/'replay/vit-dropout-native/first.pt',map_location='cpu',weights_only=False)['mask'].reshape(-1)
indices=(x!=y).nonzero().flatten()
record={'elements':x.numel(),'mismatches':indices.numel(),'first_mismatch':int(indices[0]),'last_mismatch':int(indices[-1]),'mismatches_before_786432':int((indices<786432).sum()),'reference_zeros':int((~x).sum()),'actual_zeros':int((~y).sum())}
record['rng']={}
for i in [1,2]:
 state=torch.load(r/'vit_b16-observed'/('leaf-%04d.pt'%i),map_location='cpu',weights_only=False)['rng_before']
 record['rng'][str(i)]=list(state.numpy().view('<u8').tolist())
(r/'mask-details.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
