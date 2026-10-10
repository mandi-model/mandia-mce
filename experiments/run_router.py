"""Closed-set term router. No definition or future output tokens as features.
Requires the completed extended run. Selects civil/penal only; not OOD detection.
"""
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
ROUTER_CFG={'input':'term before first newline only; no definition, expert loss, or test label in features',
 'classes':['civil','penal'],'C_grid':[.1,1.,10.],'selection':'highest validation macro F1; lower C breaks ties',
 'features':'char_wb TF-IDF ngrams 2..5; max_features 20000; min_df1; lowercase; L2 norm',
 'classifier':'balanced logistic regression, max_iter1000, seed20261010',
 'scope':'closed-set term routing for this glossary only; no general/OOD detection; no question routing claim',
 'sklearn':sklearn.__version__,'bootstrap':'2000 term-group resamples; conditional descriptive intervals'}
assert json.loads((OUT/'completion.json').read_text())['complete']
save('router_protocol.json',ROUTER_CFG)
(OUT/'router_runner.py').write_text(ROUTER_RUNNER_SOURCE);publish('router_runner.py')
split_docs={s:[r for r in DOCS if r['split']==s] for s in ['train','validation','test']}
def term(r):return r['text'].split('\n',1)[0]
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),max_features=20000,min_df=1,dtype=np.float64)
X=vec.fit_transform([term(r) for r in split_docs['train']]);yv=[r['domain'] for r in split_docs['validation']]
XV=vec.transform([term(r) for r in split_docs['validation']]);yt=[r['domain'] for r in split_docs['train']]
router_trials=[];fitted={}
for C in ROUTER_CFG['C_grid']:
 clf=LogisticRegression(C=C,class_weight='balanced',max_iter=1000,random_state=20261010).fit(X,yt)
 assert np.all(clf.n_iter_<1000), 'Router did not converge'
 pred=clf.predict(XV)
 router_trials.append({'C':C,'validation_macro_f1':float(f1_score(yv,pred,average='macro')),'validation_accuracy':float(accuracy_score(yv,pred))})
 fitted[C]=clf
C=min(router_trials,key=lambda r:(-r['validation_macro_f1'],r['C']))['C'];clf=fitted[C]
save('router_selection.json',{'selected_C':C,'trials':router_trials,'test_evaluated_at_selection':False})
state={'protocol':ROUTER_CFG,'C':C,'vocabulary':{k:int(v) for k,v in vec.vocabulary_.items()},'idf':vec.idf_.tolist(),'coef':clf.coef_[0].tolist(),'intercept':float(clf.intercept_[0]),'classes':clf.classes_.tolist()}
save('router_weights.json',state)

test_records=split_docs['test'];terms=[term(r) for r in test_records]
XT=vec.transform(terms);truth=[r['domain'] for r in test_records];pred=clf.predict(XT).tolist()
# Independently reconstruct the JSON router and verify decisions; no pickle needed.
rv=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),vocabulary=state['vocabulary'],dtype=np.float64)
rv.idf_=np.array(state['idf'])
decision=np.asarray(rv.transform(terms)@np.array(state['coef'])).ravel()+state['intercept']
restored=[state['classes'][int(z>0)] for z in decision]
assert restored==pred
max_decision_error=float(np.max(np.abs(decision-clf.decision_function(XT))))
assert max_decision_error<1e-8
test_predictions=[{'hash':r['hash'],'group':r['group'],'true_domain':y,'route':p} for r,y,p in zip(test_records,truth,pred)]
prediction_map={r['hash']:r['route'] for r in test_predictions}
groups=defaultdict(list)
for i,r in enumerate(test_records):groups[r['group']].append(i)
glist=list(groups.values());rng=np.random.default_rng(20261010);accs=[];f1s=[]
for _ in range(2000):
 indices=[i for gi in rng.integers(0,len(glist),len(glist)) for i in glist[gi]]
 a=[truth[i] for i in indices];b=[pred[i] for i in indices]
 accs.append(accuracy_score(a,b));f1s.append(f1_score(a,b,labels=['civil','penal'],average='macro',zero_division=0))
router_metrics={'test_accuracy':float(accuracy_score(truth,pred)),'test_macro_f1':float(f1_score(truth,pred,average='macro')),'majority_baseline_accuracy':max(Counter(truth).values())/len(truth),
 'confusion_matrix_civil_penal':confusion_matrix(truth,pred,labels=['civil','penal']).tolist(),'test_documents':len(truth),'term_groups':len(glist),
 'accuracy_ci95':np.quantile(accs,[.025,.975]).tolist(),'macro_f1_ci95':np.quantile(f1s,[.025,.975]).tolist(),'json_reload_max_decision_error':max_decision_error,'selected_C':C}
save('router_metrics.json',router_metrics);save('router_test_predictions.json',test_predictions)
print('ROUTEUR DE TERMES :',router_metrics,flush=True)

# End-to-end definition completion. The term is supplied input and not scored.
# Both oracle and predicted routes use identical definition/EOS targets.
DEFSTART={}
for r in test_records:
 encoding=tokenizer.encode(r['text']);boundary=len(term(r))+1
 DEFSTART[r['hash']]=next(i for i,(a,b) in enumerate(encoding.offsets) if b>boundary)
@torch.no_grad()
def score_definitions(model,method,domain,route,subset=None):
 model.eval();docs=TESTDOCS[domain] if subset is None else subset
 totals=[{'nll':0.,'targets':0} for _ in docs];chunks=[]
 for i,r in enumerate(docs):
  for start in range(0,len(r['ids'])-1,256):chunks.append((i,start,min(256,len(r['ids'])-start-1)))
 for k in range(0,len(chunks),4):
  group=chunks[k:k+4];length=max(n for i,s,n in group)
  inputs=np.full((len(group),length),eos,dtype=np.int64);labels=np.full((len(group),length),-100,dtype=np.int64)
  for bi,(i,s,n) in enumerate(group):
   r=docs[i];inputs[bi,:n]=r['ids'][s:s+n]
   ys=r['ids'][s+1:s+n+1].copy();ys[np.arange(s+1,s+n+1)<DEFSTART[r['hash']]]=-100
   labels[bi,:n]=ys
  x=torch.tensor(inputs,device='cuda');y=torch.tensor(labels,device='cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):logits=forward(model,method,x,None,route)[0]
  losses=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y.reshape(-1),ignore_index=-100,reduction='none').reshape(len(group),-1).sum(1).cpu().tolist()
  counts=(y!=-100).sum(1).cpu().tolist()
  for (i,s,n),nll,count in zip(group,losses,counts):totals[i]['nll']+=nll;totals[i]['targets']+=count
 rows=[{'hash':r['hash'],'group':r['group'],'flagged_overlap':r['hash'] in FLAGGED,**t} for r,t in zip(docs,totals)]
 return {'loss_nats':sum(r['nll'] for r in rows)/sum(r['targets'] for r in rows),'targets':sum(r['targets'] for r in rows),'documents':rows}
baseline=build('full',20261010);baseline.eval()
save('router_definition_baseline.json',{d:score_definitions(baseline,'full',d,None) for d in ['civil','penal']})
del baseline;gc.collect();torch.cuda.empty_cache()
ROUTED=[]
tests=json.loads((OUT/'test_results.json').read_text())
for order in CFG['orders']:
 for seed in CFG['evaluation_seeds']:
  for method in ['MCE','MCE64','LoRA_r8','LoRA_r32']:
   name=f'{method}_{seed}_{order[0]}_final.pt'
   ck=torch.load(OUT/name,map_location='cpu',weights_only=True)
   model=build(method,seed)
   for d in order:model.add_expert(d)
   status=model.load_state_dict(ck['state'],strict=False)
   expected={k for k in model.state_dict() if k.startswith('experts.') or '.adapters.' in k}
   assert set(ck['state'])==expected and not status.unexpected_keys
   model.eval()
   for domain in ['civil','penal']:
    oracle=score_definitions(model,method,domain,domain)
    wrong_indices=[i for i,r in enumerate(TESTDOCS[domain]) if prediction_map[r['hash']]!=domain]
    original=TESTDOCS[domain]
    if wrong_indices:
     wrong=score_definitions(model,method,domain,'penal' if domain=='civil' else 'civil',subset=[original[i] for i in wrong_indices])
     wrongmap={r['hash']:r for r in wrong['documents']}
    else:wrongmap={}
    docs=[{**d,**({'nll':wrongmap[d['hash']]['nll']} if d['hash'] in wrongmap else {}),'predicted_route':prediction_map[d['hash']]} for d in oracle['documents']]
    value=sum(d['nll'] for d in docs)/sum(d['targets'] for d in docs)
    ROUTED.append({'method':method,'seed':seed,'order':order,'domain':domain,'routed_loss_nats':value,'oracle_loss_nats':oracle['loss_nats'],'routing_penalty_nats':value-oracle['loss_nats'],'misrouted_documents':len(wrong_indices),'targets':oracle['targets'],'documents':docs,'evaluation':'definition and EOS targets only; supplied term tokens not scored; no OOD/general detector'})
   save('routed_test_results.json',ROUTED)
   del model,ck;gc.collect();torch.cuda.empty_cache()
   print('Routage mesuré',method,seed,'/'.join(order),flush=True)
save('router_completion.json',{'complete':True,'models_evaluated':24,'legal_result_rows':len(ROUTED),'no_test_based_training_or_selection':True,'term_router_only':True})
manifest()
print('ROUTEUR APPRENTI ET ÉVALUATION DES 24 BANQUES TERMINÉS',flush=True)
