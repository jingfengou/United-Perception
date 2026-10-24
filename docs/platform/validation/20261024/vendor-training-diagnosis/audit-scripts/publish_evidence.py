from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

repo=Path.cwd()
private=repo/'dist/vendor-training-diagnosis-20261024'
destination=repo/'docs/platform/validation/20261024/vendor-training-diagnosis'
protocol=json.loads((destination/'protocol.json').read_text())
old=repo/'dist/metax-deterministic-train-20261024'

def checksum(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_bytes((json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def close(reference,actual):return abs(actual-reference)<=1e-5+1e-4*abs(reference)

summary={'scope':'First-step and bounded-prefix diagnosis; no new full-budget accuracy claim',
         'diagnostic_tools_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
         'production_components_sha256':protocol['production_components_sha256'],
         'old_multiseed_status':'paused','optimizer_updates':0,'vendors':{},'observed_prefix_checks':{},'receipts':{}}
expected_archives={'nvidia':'49eb8313893e9ae62a30f6b36aefd6cb2ce0ba2c5994e8cf7b5c09e864673b62',
                   'metax':'bff70b8d3c4310faf36e04c37ca703d76c276159ea3a0768be45cb33c99c2c7d'}

for vendor in ('nvidia','metax'):
 source=private/(vendor+'-collected02')
 index=read(source/'evidence-index.json')
 assert checksum(private/(vendor+'-collected02.tar.gz'))==expected_archives[vendor]
 def verified(relative):
  path=source/relative
  info=index['files'][relative]
  assert checksum(path)==info['sha256'] and path.stat().st_size==info['bytes'],relative
  return path
 def copy(relative,public):
  target=destination/public;target.parent.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(verified(relative),target)

 shutil.copyfile(source/'evidence-index.json',destination/(vendor+'-evidence-index.json'))
 copy('postflight02.json',vendor+'-postflight.json')
 post=read(source/'postflight02.json')
 assert post['code_and_data_intact'] and post['lock_available_for_reuse']
 assert post['production_components_sha256']==protocol['production_components_sha256']
 copy('counter-details.json',vendor+'-counter-details.json')
 transfer=private/(vendor+'-transfer.json')
 transfer_result=read(transfer)
 assert not transfer_result['up_imported'] and not transfer_result['up_platform_imported']
 assert all(v['values_exact'] for v in transfer_result['copies'].values())
 shutil.copyfile(transfer,destination/(vendor+'-transfer.json'))
 cpu='nvidia-cpu03.xml' if vendor=='nvidia' else 'metax-cpu02.xml'
 copy(cpu,vendor+'-cpu.xml')
 suite=ET.parse(source/cpu).getroot().find('testsuite').attrib
 assert suite['tests']=='81' and all(suite[k]=='0' for k in ('failures','errors','skipped'))
 copy(('pilot' if vendor=='nvidia' else 'pilot02')+'/report.json','pilots/'+vendor+'.json')

 runs={}
 folders=[model+'-'+variant for model in ('vit_b16','yolov5s') for variant in ('observed','control')]
 if vendor=='nvidia':folders += [m+'-original' for m in ('vit_b16','yolov5s')]
 folders += ['writes','writes02','yolo-deep']
 for folder in folders:
  relative=folder+'/report.json';r=read(verified(relative))
  model=r['model'];expected=protocol['models'][model]
  assert r['status']=='probe_completed' and r['seed']==131 and r['mode']=='native_deterministic'
  for key in ('initial_file_sha256','initial_state_sha256','algorithm_config_sha256','driver_sha256'):
   assert r[key]==expected[key],(vendor,folder,key)
  assert r['data_sha256']==protocol['data_sha256']
  assert r['restore_weights_exact'] and r['checkpoint_all_states_finite']
  env=r['environment']
  assert env['deterministic'] and env['cudnn_deterministic'] and not env['deterministic_warn_only']
  assert not env['cudnn_benchmark'] and not env['matmul_tf32'] and not env['cudnn_tf32']
  keys=('status','model','lane','seed','mode','training_steps','initial_file_sha256','initial_state_sha256',
        'algorithm_config_sha256','driver_sha256','data_sha256','first_batch_sha256','all_batches_sha256',
        'losses','learning_rates','restore_weights_exact','checkpoint_all_states_finite','environment')
  selected={k:r[k] for k in keys}
  selected.update(scope='Diagnostic prefix only; final accuracy omitted; raw report retained',
                  raw_report={'relative_path':relative,**index['files'][relative]},
                  frozen_source_sha256=r['source']['sha256'])
  write(destination/'runs'/vendor/(folder+'.json'),selected)
  runs[folder]=selected
  summary['optimizer_updates']+=r['training_steps']
 for model in ('vit_b16','yolov5s'):
  copy(model+'-observed/observations.json','observations/'+vendor+'/'+model+'.json')
  prefix=runs[model+'-control']
  path=old/('nvidia/reference/'+model+'/new-01/report.json' if vendor=='nvidia' else 'collected/results/'+model+'/report.json')
  full=read(path)
  assert prefix['initial_file_sha256']==full['initial_file_sha256']
  assert prefix['first_batch_sha256']==full['first_batch_sha256']
  values=full['losses'][:5]
  differences=[a-b for a,b in zip(prefix['losses'],values)]
  record={'prefix_losses':prefix['losses'],'full_training_reference_losses':values,
          'reference_report_sha256':checksum(path),'deltas':differences,
          'exact':prefix['losses']==values,'all_within_tolerance':all(close(x,y) for x,y in zip(values,prefix['losses'])),
          'first_loss_exact':prefix['losses'][0]==values[0]}
  assert record['first_loss_exact']
  summary['observed_prefix_checks'][vendor+'/'+model]=record
 copy('yolo-deep/observations.json','observations/'+vendor+'/yolo-deep.json')
 replays=[]
 for relative in sorted(name for name in index['files'] if (name.startswith('replay/') or name.startswith('first-divergence/')) and name.endswith('/report.json')):
  r=read(verified(relative));assert not r['up_imported'] and not r['up_platform_imported']
  assert len(r['repeats'])==10
  first=r['repeats'][0]['fingerprint']
  changed=[[key for key in first if first[key]!=record['fingerprint'][key]] for record in r['repeats']]
  selected={key:r[key] for key in ('scope','torch','device','repr','layout','all_repeats_exact','common_capture_sha256','own_rng_capture_sha256','up_imported','up_platform_imported')}
  selected.update(raw_report={'relative_path':relative,**index['files'][relative]},
                  repeats=10,changed_fields_by_repetition=changed,
                  first_record=r['repeats'][0])
  replays.append(selected)
 write(destination/(vendor+'-replays.json'),replays)
 copy('writes-replay/report.json',vendor+'-writes-replay.json')
 writes=read(verified('writes02/writes.json'))
 assert not writes['gpu_random_calls']
 write(destination/(vendor+'-writes-audit.json'),{
  'scope':'Corrected counting includes GPU device-only creators, plus Tensor operands',
  'recorded_writes':len(writes['records']),'gpu_random_calls':writes['gpu_random_calls'],
  'observer_sha256':writes['observer_sha256'],'raw_report':index['files']['writes02/writes.json']})
 summary['vendors'][vendor]={'cpu_tests':81,'optimizer_updates':sum(r['training_steps'] for r in runs.values()),
   'raw_files':len(index['files']),'retained_tensor_bytes':sum(r['bytes'] for name,r in index['files'].items() if name.endswith('.pt')),
   'jobs':{name:{key:row[key] for key in ('status','exit_code','script_sha256','device','started_utc','finished_utc')} for name,row in index['jobs'].items()},
   'prefix_losses':{name:r['losses'] for name,r in runs.items()}}
 summary['receipts'][vendor]={'archive':'collected02.tar.gz','bytes':(private/(vendor+'-collected02.tar.gz')).stat().st_size,
   'sha256':expected_archives[vendor],'root':index['root'],'publicly_downloadable':False,
   'evidence_index_sha256':checksum(source/'evidence-index.json'),
   'supplemental_transfer_result':{'raw_relative_path':'transfer.json.result','sha256':checksum(transfer)}}

for relative,name in [('controls-comparison.json','controls.json'),('vit_b16-comparison.json','vit_b16.json'),('yolov5s-comparison.json','yolov5s.json')]:
 source=private/'nvidia-collected02';index=read(source/'evidence-index.json')
 assert checksum(source/relative)==index['files'][relative]['sha256']
 target=destination/'comparisons'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source/relative,target)
for name in ('yolo32-comparison.json','replay-comparison.json','bn14-comparison.json','bn-fp64-reference.json','mask-details.json'):
 source=private/'metax-collected02';index=read(source/'evidence-index.json')
 assert checksum(source/name)==index['files'][name]['sha256']
 shutil.copyfile(source/name,destination/name)
assert summary['optimizer_updates']==protocol['total_optimizer_updates']==48
summary['tolerance']={'rtol':1e-4,'atol':1e-5}
summary['runtime_files_changed']=subprocess.check_output(['git','diff','--name-only',
 'b137c68a1988e99b3cb19ac3307796b9ab70a984','--','up','up_platform','configs'],text=True).splitlines()
assert not summary['runtime_files_changed']
summary['limitations']=['No new complete-budget accuracy or multiseed equivalence test',
 'C500 YOLO control step5 does not match the prior full run within tolerance; record synchronization/repeatability limitation',
 'First differing BN has already different inputs; common-input forward stays within tolerance',
 'FP64 diagnostic shows both vendors FP32 BN reduction errors; NVIDIA is not an exact arithmetic oracle',
 'Vendor closed-source kernel implementation not audited; measured RNG mapping matches public CUDA policy']
summary['retained_failed_attempts']=[{'vendor':'metax','stage':'pilot','reason':'Missing USER/LOGNAME for UID1009; import failed before numerical execution'},
 {'vendor':'nvidia','stage':'nvidia-cpu.xml','reason':'Missing PYTHONPATH during collection; preserved 6 collection errors; final 81 tests pass'},
 {'stage':'CPU comparison before complete leaf exchange','reason':'Explicitly bounded the comparison to transferred leaves; no partial data accepted'},
 {'stage':'incomplete forward archive transfer','reason':'Retry has full size and SHA256 00942861aab7b6c2c387fd52395a827b9cc810d3da443f487a0ed8af14265270'}]
write(destination/'summary.json',summary)
script_archive={}
for vendor,names in {'nvidia':['compare_controls.py'],
                     'metax':['bn_oracle.py','counter_details.py','compare_bn14.py','compare_replay.py','mask_details.py']}.items():
 source=private/(vendor+'-collected02');index=read(source/'evidence-index.json')
 for name in names:
  assert checksum(source/name)==index['files'][name]['sha256']
  target=destination/'audit-scripts'/name;target.parent.mkdir(exist_ok=True)
  target.write_bytes((source/name).read_bytes().replace(b'\r\n',b'\n'))
  script_archive[name]={'vendor':vendor,'raw_relative_path':name,
    'executed_sha256':checksum(source/name),'archived_sha256':checksum(target),
    'normalization':'CRLF to LF only'}
shutil.copyfile(Path(__file__),destination/'audit-scripts/publish_evidence.py')
write(destination/'audit-scripts/index.json',script_archive)
publication={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':summary['scope'],
  'optimizer_updates':48,'old_multiseed_status':'paused','files':{}}
for path in sorted(destination.rglob('*')):
 if path.is_file() and path.name!='publication-index.json':
  publication['files'][path.relative_to(destination).as_posix()]={'bytes':path.stat().st_size,'sha256':checksum(path)}
write(destination/'publication-index.json',publication)
print(json.dumps({'public_files':len(publication['files'])+1,'optimizer_updates':48,
                  'public_bytes':sum(row['bytes'] for row in publication['files'].values()),
                  'prefix_checks':{name:row['all_within_tolerance'] for name,row in summary['observed_prefix_checks'].items()}},indent=2))
