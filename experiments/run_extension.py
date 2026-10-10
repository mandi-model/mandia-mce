# Mandi: fixed, exploratory extension. No test-guided model selection.
import os, sys, json, hashlib, shutil, gc, time, math, re, unicodedata
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import numpy as np
import torch
from tokenizers import Tokenizer

# Public launcher: provide archived inputs through environment variables.
# The experiment logic below is preserved; cloud mounting and private paths are removed.
OLD=Path(os.environ['MCE_INPUT_DIR']).resolve()
PARENT=Path(os.environ['MCE_CORE_DIR']).resolve()
CORPUS=Path(os.environ['MCE_CORPUS_DIR']).resolve()
assert torch.cuda.is_available(), 'CUDA GPU required for this historical protocol'
RUN='legalextended_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
OUTPUT_ROOT=Path(os.environ.get('MCE_OUTPUT_DIR','runs')).resolve()
OUT=OUTPUT_ROOT/RUN;OUT.mkdir(parents=True,exist_ok=False)
DEST=OUTPUT_ROOT/(RUN+'_archive');DEST.mkdir(parents=True,exist_ok=False)
T0=time.perf_counter()
CFG={'scope':'exploratory extension of one legal glossary; not the four-domain confirmatory study',
     'methods':['full','MCE','MCE64','LoRA_r8','LoRA_r32'],
     'lr_grid':[1e-4,3e-4,1e-3], 'tuning_seed':20261012,
     'evaluation_seeds':[20261013,20261014,20261015],
     'orders':[['civil','penal'],['penal','civil']],
     'tuning_order':['civil','penal'],
     'selection':'minimum unweighted mean of final civil and penal validation losses at fixed step 300; lower LR breaks ties; no general/test selection',
     'steps_per_domain':300,'batch_size':4,'sequence_length':256,'warmup_steps':30,
     'validation_curve_steps':[100,200,300],'tokens_per_domain':307200,
     'maximum_walltime_seconds':3600,'optimizer':'AdamW .9/.95; matrix decay .1; clip 1; reset per domain',
     'checkpoint':'fixed final 300, no checkpoint selection',
     'source':'bernard-ng/french-legal-glossary','license_declared':'CC BY-NC-SA 4.0; private study',
     'core_dropout':'full core train mode; all frozen cores eval mode; adapters dropout .1',
     'test_protocol':'LR frozen after all tuning, then final models trained under fixed script; no selection or rerun from test results',
     'precision':'bfloat16 autocast, FP32 weights','torch':str(torch.__version__),
     'gpu':torch.cuda.get_device_name(0),'native_bf16':torch.cuda.is_bf16_supported(including_emulation=False)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def publish(name):
 temp=DEST/(name+'.partial');shutil.copy2(OUT/name,temp)
 assert sha(temp)==sha(OUT/name)
 temp.replace(DEST/name)
def save(name,obj):
 (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');publish(name)
def manifest():
 save('manifest.json',{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.iterdir() if p.is_file() and p.name not in ('manifest.json','core_checkpoint.pt')})
def budget():
 if time.perf_counter()-T0>CFG['maximum_walltime_seconds']:raise RuntimeError('Bounded extension budget exceeded; completed runs saved')
old_manifest=json.loads((OLD/'manifest.json').read_text())
for name in ['mandi_model.py','mandi_mce_v1.py','tokenizer.json','documents.jsonl','general_validation.bin']+[d+'_'+s+'.bin' for d in ['civil','penal'] for s in ['train','validation','test']]:
 shutil.copy2(OLD/name,OUT/name)
 assert sha(OUT/name)==old_manifest[name]['sha256'],name
 publish(name)
shutil.copy2(PARENT/'latest.pt',OUT/'core_checkpoint.pt')
assert sha(OUT/'core_checkpoint.pt')=='38233f63ee915a882ee49f0bbf79b280389101ab46f2d09d27ad2f08559e43ab'
CFG['core_sha256']=sha(OUT/'core_checkpoint.pt');CFG['previous_pilot']=OLD.name
sys.path.insert(0,str(OUT))
from mandi_model import MandiLM
from mandi_mce_v1 import MandiMCE,MandiLoRABank
loaded=torch.load(OUT/'core_checkpoint.pt',map_location='cpu',weights_only=True)
BASE=loaded['model'];MODEL_CFG=loaded['config'];del loaded
tokenizer=Tokenizer.from_file(str(OUT/'tokenizer.json'))
eos=tokenizer.token_to_id('<eos>')
TRAIN={d:np.memmap(OUT/(d+'_train.bin'),dtype='<u2',mode='r') for d in ['civil','penal']}
VAL={d:np.memmap(OUT/(d+'_validation.bin'),dtype='<u2',mode='r') for d in ['civil','penal']}
GENERAL=np.memmap(OUT/'general_validation.bin',dtype='<u2',mode='r')
GSTART=json.loads((OLD/'config.json').read_text())['general_validation_starts']
DOCS=[json.loads(line) for line in (OUT/'documents.jsonl').read_text().splitlines()]
CFG['data_manifest_sha256']=sha(OLD/'manifest.json')
CFG['tokenizer_sha256']=sha(OUT/'tokenizer.json')
CFG['source_sha256']='c26153be965d4ef30c6125ad4770827d0a91d5ce1851607092e33ab28ac5221d'
save('config.json',CFG)
(OUT/'runner.py').write_text(EXTENDED_RUNNER_SOURCE);publish('runner.py')
(OUT/'protocol.md').write_text(EXTENDED_PROTOCOL_SOURCE);publish('protocol.md')
manifest()
print('EXTENSION INITIALISÉE',RUN,flush=True)

# Audit partial lexical contamination against general training only. Test texts
# are used for this fixed audit, never as optimization targets or LR selection.
def grams(text):
 words=re.findall(r'\w+',unicodedata.normalize('NFC',text).casefold())
 return set(tuple(words[i:i+5]) for i in range(len(words)-4))
legal_grams=[grams(r['text'].split('\n',1)[1]) for r in DOCS]
index=defaultdict(list)
for i,ss in enumerate(legal_grams):
 for g in ss:index[g].append(i)
maximum=[{'matched_5grams':0,'containment':0.,'general_train_hash':None} for _ in DOCS]
scanned=0
with (CORPUS/'train.jsonl').open() as f:
 for line in f:
  record=json.loads(line);ss=grams(record['text'])
  counts=Counter(i for g in ss if g in index for i in index[g])
  for i,n in counts.items():
   rate=n/len(legal_grams[i])
   if rate>maximum[i]['containment']:maximum[i]={'matched_5grams':n,'containment':rate,'general_train_hash':record['hash']}
  scanned+=1
  if scanned%5000==0:print('Audit lexical',scanned,'documents généraux',flush=True);budget()
audit=[]
for r,v in zip(DOCS,maximum):
 audit.append({'hash':r['hash'],'domain':r['domain'],'split':r['split'],'group':r['group'],**v,
               'flagged':v['containment']>=.5 and v['matched_5grams']>=20})
save('lexical_overlap_audit.json',{'general_training_documents_scanned':scanned,'threshold':'containment >=.5 and >=20 unique shared word 5-grams in one general training document',
     'semantic_overlap_audited':False,'training_data_modified':False,'records':audit})
FLAGGED={r['hash'] for r in audit if r['flagged']}
del index,legal_grams,maximum;gc.collect()
print('Audit terminé :',len(FLAGGED),'définitions signalées; pas de suppression post-hoc du train.',flush=True)

SEEDS=[CFG['tuning_seed']]+CFG['evaluation_seeds']
WINDOWS={str(seed):{d:np.random.default_rng(seed+(0 if d=='civil' else 1)).integers(0,len(TRAIN[d])-257,size=(300,4)).tolist() for d in ['civil','penal']} for seed in SEEDS}
save('training_windows.json',WINDOWS)
def batch(array,starts,length=256):
 x=torch.tensor(np.stack([array[s:s+length].astype(np.int64) for s in starts]),device='cuda')
 y=torch.tensor(np.stack([array[s+1:s+length+1].astype(np.int64) for s in starts]),device='cuda')
 return x,y
def forward(model,method,x,y=None,route=None):
 return model(x,y) if method=='full' else model(x,y,expert=route)
def phash(named):
 h=hashlib.sha256()
 for n,p in sorted(named):h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
@torch.no_grad()
def evaluate(model,method,array,route=None,starts=None,details=False):
 model.eval();positions=list(range(0,len(array)-1,256)) if starts is None else starts
 weighted=0.;total=0;rows=[]
 for k in range(0,len(positions),4):
  group=positions[k:k+4];full=[s for s in group if len(array)-s-1>=256]
  if full:
   x,y=batch(array,full)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=forward(model,method,x,None,route)[0]
   losses=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').reshape(len(full),-1).sum(1).cpu().tolist()
   for s,nll in zip(full,losses):weighted+=nll;total+=256;rows.append({'start':int(s),'nll':nll,'targets':256})
  for s in group:
   n=len(array)-s-1
   if n<256:
    x,y=batch(array,[s],n)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=forward(model,method,x,None,route)[0]
    nll=float(torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum'))
    weighted+=nll;total+=n;rows.append({'start':int(s),'nll':nll,'targets':n})
 assert total==(len(array)-1 if starts is None else len(starts)*256)
 return {'loss_nats':weighted/total,'targets':total,**({'windows':rows} if details else {})}
TESTDOCS=None;GTEST=None;GTSTART=None
@torch.no_grad()
def test_documents(model,method,domain,route=None):
 # Batch equal-length complete windows across documents; preserve doc-level NLL.
 docs=TESTDOCS[domain];totals=[{'nll':0.,'targets':0} for _ in docs];chunks=[]
 for i,r in enumerate(docs):
  a=r['ids']
  for s in range(0,len(a)-1,256):chunks.append((i,s,min(256,len(a)-s-1)))
 complete=[r for r in chunks if r[2]==256]
 for k in range(0,len(complete),4):
  group=complete[k:k+4]
  x=torch.tensor(np.stack([docs[i]['ids'][s:s+256] for i,s,n in group]),dtype=torch.long,device='cuda')
  y=torch.tensor(np.stack([docs[i]['ids'][s+1:s+257] for i,s,n in group]),dtype=torch.long,device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=forward(model,method,x,None,route)[0]
  losses=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').reshape(len(group),-1).sum(1).cpu().tolist()
  for (i,s,n),nll in zip(group,losses):totals[i]['nll']+=nll;totals[i]['targets']+=n
 for i,s,n in chunks:
  if n==256:continue
  x=torch.tensor([docs[i]['ids'][s:s+n]],device='cuda');y=torch.tensor([docs[i]['ids'][s+1:s+n+1]],device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=forward(model,method,x,None,route)[0]
  nll=float(torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='sum'))
  totals[i]['nll']+=nll;totals[i]['targets']+=n
 rows=[{'hash':r['hash'],'group':r['group'],'flagged_overlap':r['hash'] in FLAGGED,**t} for r,t in zip(docs,totals)]
 return {'loss_nats':sum(r['nll'] for r in rows)/sum(r['targets'] for r in rows),'targets':sum(r['targets'] for r in rows),'documents':rows}
CURVES=[];COSTS=[];CONTROLS=[];TUNING=[];MATRIX=[];TEST=[];RELOAD=[];LOG=[]
def build(method,seed):
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 core=MandiLM(MODEL_CFG).to('cuda');core.load_state_dict(BASE,strict=True)
 return core if method=='full' else MandiMCE(core,bottleneck=64 if method=='MCE64' else 256) if method.startswith('MCE') else MandiLoRABank(core,rank=8 if method=='LoRA_r8' else 32)
def run(method,seed,order,lr,phase):
 budget();model=build(method,seed);introduced=[]
 key=f'{phase}_{method}_{seed}_{order[0]}_{lr:.0e}'
 corehash=None if method=='full' else phash(model.frozen_core_parameters())
 def scores(stage,step):
  values={}
  for d in ['general','civil','penal']:
   route=d if d in introduced else None
   result=evaluate(model,method,GENERAL if d=='general' else VAL[d],route,GSTART if d=='general' else None)
   row={'run':key,'phase':phase,'method':method,'seed':seed,'order':order,'lr':lr,'stage':stage,'step':step,'domain':d,**result}
   CURVES.append(row);values[d]=result['loss_nats']
   if step in (0,300):MATRIX.append(row)
  save('validation_curves.json',CURVES)
  return values
 scores(0,0)
 oldhash=None;oldprobe=None
 for stage,domain in enumerate(order,1):
  # Domain-specific seed makes adapter initialization and sampling order invariant.
  domainseed=seed+(1000 if domain=='civil' else 2000)
  torch.manual_seed(domainseed);torch.cuda.manual_seed_all(domainseed)
  parameters=list(model.parameters()) if method=='full' else model.add_expert(domain)
  model.train();introduced.append(domain)
  decay=[p for p in parameters if p.ndim>=2];nodecay=[p for p in parameters if p.ndim<2]
  opt=torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':nodecay,'weight_decay':0.}],lr=lr,betas=(.9,.95))
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
  elapsed=0.;peak=0
  for step,starts in enumerate(WINDOWS[str(seed)][domain],1):
   budget();model.train();torch.cuda.synchronize();t=time.perf_counter()
   x,y=batch(TRAIN[domain],starts);warm=min(1.,step/30);progress=max(0.,(step-30)/270)
   rate=lr*warm*(.1+.9*(1+math.cos(math.pi*progress))/2)
   for g in opt.param_groups:g['lr']=rate
   model.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):logits,loss=forward(model,method,x,y,domain)
   assert torch.isfinite(loss);loss.backward()
   if method!='full':
    assert all(p.grad is None for _,p in model.frozen_core_parameters())
    if stage==2:assert all(p.grad is None for p in model.domain_parameters(order[0]))
   torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);opt.step()
   torch.cuda.synchronize();elapsed+=time.perf_counter()-t;peak=max(peak,torch.cuda.max_memory_allocated())
   if step%25==0:LOG.append({'run':key,'domain':domain,'step':step,'train_loss':float(loss),'lr':rate})
   if step in CFG['validation_curve_steps']:
    current=scores(stage,step)
    print(phase,method,seed,'/'.join(order),'LR',lr,domain,step,{d:round(v,4) for d,v in current.items()},flush=True)
  model.zero_grad(set_to_none=True);del opt,x,y,logits,loss;gc.collect()
  COSTS.append({'run':key,'phase':phase,'method':method,'seed':seed,'order':order,'domain':domain,'steps':300,'tokens':307200,'trainable_parameters':sum(p.numel() for p in parameters),'training_seconds':elapsed,'peak_gpu_bytes':peak})
  if method!='full':
   assert phash(model.frozen_core_parameters())==corehash
   probe,_=batch(VAL[order[0]],[0],32)
   with torch.no_grad():newprobe=forward(model,method,probe,None,order[0])[0].detach().clone()
   if stage==2:
    assert phash(model.named_domain_parameters(order[0]))==oldhash
    assert torch.equal(newprobe,oldprobe)
   oldhash=phash(model.named_domain_parameters(order[0]));oldprobe=newprobe
  CONTROLS.append({'run':key,'stage':stage,'frozen_core_verified':method!='full','old_route_unchanged':method!='full' and stage==2})
  for name,obj in [('costs.json',COSTS),('controls.json',CONTROLS),('training_log.json',LOG),('validation_matrix.json',MATRIX)]:save(name,obj)
 if phase=='tuning':
  TUNING.append({'method':method,'seed':seed,'lr':lr,'validation_mean_legal':(current['civil']+current['penal'])/2,'final_losses':current})
  save('tuning_results.json',TUNING)
 else:
  model.eval()
  for d in ['general','civil','penal']:
   result=evaluate(model,method,GTEST,None,GTSTART,details=True) if d=='general' else test_documents(model,method,d,d)
   TEST.append({'run':key,'method':method,'seed':seed,'order':order,'lr':lr,'domain':d,**result})
  save('test_results.json',TEST)
  state=model.state_dict()
  if method.startswith('MCE'):state={k:v for k,v in state.items() if k.startswith('experts.')}
  elif method.startswith('LoRA'):state={k:v for k,v in state.items() if '.adapters.' in k}
  # Preserve shared storage for tied embeddings/head in full checkpoints.
  cpustate={};cache={}
  for k,v in state.items():
   storage=(v.untyped_storage().data_ptr(),v.storage_offset(),tuple(v.shape),tuple(v.stride()))
   if storage not in cache:cache[storage]=v.detach().cpu().clone()
   cpustate[k]=cache[storage]
  name=f'{method}_{seed}_{order[0]}_final.pt'
  ck={'method':method,'seed':seed,'domains':order,'order':order,'lr':lr,'core_checkpoint_sha256':CFG['core_sha256'],'core_config':MODEL_CFG,'state':cpustate,'bottleneck':64 if method=='MCE64' else 256,'rank':8 if method=='LoRA_r8' else 32 if method=='LoRA_r32' else None,'scope':CFG['scope']}
  torch.save(ck,OUT/name)
  reread=torch.load(OUT/name,map_location='cpu',weights_only=True)
  assert all(torch.equal(reread['state'][k],v) for k,v in cpustate.items())
  del model;gc.collect();torch.cuda.empty_cache()
  reconstructed=build(method,seed)
  if method!='full':
   for d in order:reconstructed.add_expert(d)
   actual=reconstructed.load_state_dict(reread['state'],strict=False)
   expected={k for k in reconstructed.state_dict() if k.startswith('experts.') or '.adapters.' in k}
   assert set(reread['state'])==expected and not actual.unexpected_keys
  else:reconstructed.load_state_dict(reread['state'],strict=True)
  reconstructed.eval();maxerr=0.
  for d in ['general','civil','penal']:
   score=evaluate(reconstructed,method,GENERAL if d=='general' else VAL[d],None if d=='general' else d,GSTART if d=='general' else None)['loss_nats']
   maxerr=max(maxerr,abs(score-current[d]))
  assert maxerr<1e-6
  RELOAD.append({'run':key,'checkpoint':name,'max_validation_error':maxerr,'sha256':sha(OUT/name)})
  save('reload_checks.json',RELOAD);publish(name)
  del reconstructed,reread,ck,cpustate,state,cache
  model=None
 del model,parameters,decay,nodecay,oldprobe
 gc.collect();torch.cuda.empty_cache();manifest()

for method in CFG['methods']:
 for lr in CFG['lr_grid']:run(method,CFG['tuning_seed'],CFG['tuning_order'],lr,'tuning')
SELECTED={m:min([r for r in TUNING if r['method']==m],key=lambda r:(r['validation_mean_legal'],r['lr']))['lr'] for m in CFG['methods']}
save('selected_hyperparameters.json',{'selected_lr':SELECTED,'selection':CFG['selection'],'tuning_complete':True,'test_evaluated_at_selection':False})
CFG['selection_sha256']=sha(OUT/'selected_hyperparameters.json');CFG['test_evaluation_started_after_selection']=True
save('config.json',CFG);manifest()
print('RÉGLAGES FIGÉS AVANT LE TEST',SELECTED,flush=True)

# Test arrays become available to model evaluation only after LR selection.
TESTDOCS={d:[] for d in ['civil','penal']}
for r in DOCS:
 if r['split']=='test':TESTDOCS[r['domain']].append({**r,'ids':np.array(tokenizer.encode(r['text']).ids+[eos],dtype=np.int64)})
shutil.copy2(CORPUS/'test.bin',OUT/'general_test.bin')
corpus_manifest=json.loads((CORPUS/'manifest.json').read_text())
assert sha(OUT/'general_test.bin')==corpus_manifest['test.bin']['sha256'];publish('general_test.bin')
GTEST=np.memmap(OUT/'general_test.bin',dtype='<u2',mode='r');GTSTART=np.linspace(0,len(GTEST)-257,32,dtype=int).tolist()
save('test_protocol.json',{'general_test_sha256':sha(OUT/'general_test.bin'),'general_fixed_starts':GTSTART,'context':256,'general_targets':8192,'legal_evaluation':'per-document windows; no transitions between documents; EOS predicted once; first token of each document not predicted','legal_documents':{d:len(v) for d,v in TESTDOCS.items()},'selection_sha256':CFG['selection_sha256'],'lexical_sensitivity_exclusion':'flagged test documents excluded only from separately labeled sensitivity estimates; all primary test documents retained'})
baseline=build('full',20261012);baseline.eval()
save('test_baseline.json',{d:(evaluate(baseline,'full',GTEST,None,GTSTART,details=True) if d=='general' else test_documents(baseline,'full',d)) for d in ['general','civil','penal']})
del baseline;gc.collect();torch.cuda.empty_cache()
for order in CFG['orders']:
 for si,seed in enumerate(CFG['evaluation_seeds']):
  # Rotate method order to reduce the fixed-order timing bias of the first pilot.
  methods=CFG['methods'][si:]+CFG['methods'][:si]
  for method in methods:run(method,seed,order,SELECTED[method],'evaluation')
save('completion.json',{'complete':True,'run':RUN,'elapsed_seconds':time.perf_counter()-T0,'tuning_models':15,'evaluation_models':30,'trained_tokens_total':45*2*307200,'selected_lr':SELECTED,'test_evaluated':True,'reload_models':len(RELOAD),'maximum_reload_error':max(r['max_validation_error'] for r in RELOAD),'private':True,'confirmatory_four_domain_study_complete':False})
manifest()
print('ÉTUDE APPROFONDIE TERMINÉE : 5 MÉTHODES, 2 ORDRES, 3 GRAINES, TEST RÉSERVÉ ÉVALUÉ',flush=True)
print(DEST,flush=True)
