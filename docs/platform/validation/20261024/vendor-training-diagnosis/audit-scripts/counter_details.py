import json
from pathlib import Path

import numpy as np
import torch

root=Path(__file__).parent
states={}
for index in (1,2):
 payload=torch.load(root/'vit_b16-observed'/('leaf-%04d.pt'%index),map_location='cpu',weights_only=False)
 raw=payload['rng_before'].numpy().tobytes()
 states[str(index)]={'raw_hex':raw.hex(),'uint64_values':np.frombuffer(raw,dtype='<u8').tolist()}
properties=json.loads((root/'device-metadata.json').read_text())
blocks=properties['multi_processor_count']*(properties['max_threads_per_multi_processor']//256)
result={'scope':'Observed native RNG states plus comparison with PyTorch 2.10 CUDA Dropout launch formula',
        'rng_states_before_leaf':states,'properties':properties,'block_size':256,'vector_width':4,
        'property_limited_blocks':blocks,'first_vector_pass_capacity':blocks*256*4,
        'token_elements':64*17*768,'predicted_counter_increment':((64*17*768-1)//(blocks*256*4)+1)*4}
(root/'counter-details.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
