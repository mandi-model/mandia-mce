"""Compute tables, descriptive clustered bootstrap and plots from observed files.
Usage: python analyser_extension.py RAW_DIRECTORY OUTPUT_DIRECTORY
Never manufacture missing results or select models from test.
"""
from pathlib import Path
import sys,json,csv,statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

raw=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((raw/n).read_text())
done=read('completion.json');cfg=read('config.json');tests=read('test_results.json');curves=read('validation_curves.json');costs=read('costs.json');reloads=read('reload_checks.json');audit=read('lexical_overlap_audit.json');baseline=read('test_baseline.json');tuning=read('tuning_results.json');selection=read('selected_hyperparameters.json')
assert done['complete'] and len(tests)==90 and len(reloads)==30 and len(tuning)==15
assert len(curves)==945 and len(costs)==90
assert not selection['test_evaluated_at_selection']
assert all(r['max_validation_error']<1e-6 for r in reloads)
METHODS=cfg['methods'];ORDERS=cfg['orders'];SEEDS=cfg['evaluation_seeds']
LABEL={'full':'Complet','MCE':'MCE 256','MCE64':'MCE 64','LoRA_r8':'LoRA r8','LoRA_r32':'LoRA r32'}
COLORS={'full':'#b44a3a','MCE':'#126e76','MCE64':'#599e91','LoRA_r8':'#7c6ca7','LoRA_r32':'#c49321'}
def ms(a):return {'mean':st.mean(a),'sd':st.stdev(a)}
def fmt(x):return f"{x['mean']:.3f} ± {x['sd']:.3f}"
summaries=[]
for order in ORDERS:
 for method in METHODS:
  rows=[r for r in tests if r['method']==method and r['order']==order]
  assert len(rows)==9
  result={'order':order,'method':method,'lr':selection['selected_lr'][method]}
  for domain in ['general','civil','penal']:
   rs=[r for r in rows if r['domain']==domain]
   assert {r['seed'] for r in rs}==set(SEEDS)
   result[domain+'_test']=ms([r['loss_nats'] for r in rs])
   if domain!='general':
    clean=[sum(d['nll'] for d in r['documents'] if not d['flagged_overlap'])/sum(d['targets'] for d in r['documents'] if not d['flagged_overlap']) for r in rs]
    result[domain+'_test_lexical_sensitivity']=ms(clean)
  forgotten=[];general_drift=[]
  for seed in SEEDS:
   c=[r for r in curves if r['phase']=='evaluation' and r['method']==method and r['seed']==seed and r['order']==order]
   first=[r['loss_nats'] for r in c if r['domain']==order[0] and r['stage']==1 and r['step']==300][0]
   final=[r['loss_nats'] for r in c if r['domain']==order[0] and r['stage']==2 and r['step']==300][0]
   forgotten.append(max(0,final-first))
   g0=[r['loss_nats'] for r in c if r['domain']=='general' and r['stage']==0][0]
   gf=[r['loss_nats'] for r in c if r['domain']=='general' and r['stage']==2 and r['step']==300][0]
   general_drift.append(gf-g0)
  result['first_domain_forgetting_validation']=ms(forgotten);result['general_drift_validation']=ms(general_drift)
  cs=[r for r in costs if r['phase']=='evaluation' and r['method']==method and r['order']==order]
  result['trainable_parameters_per_domain']=cs[0]['trainable_parameters']
  result['training_seconds_two_domains']=ms([sum(r['training_seconds'] for r in cs if r['seed']==s) for s in SEEDS])
  result['peak_gpu_GB']=max(r['peak_gpu_bytes'] for r in cs)/1e9
  summaries.append(result)

paired=[]
for order in ORDERS:
 for domain in ['civil','penal']:
  for a,b in [('MCE','LoRA_r32'),('MCE64','LoRA_r8')]:
   maps={}
   for method in [a,b]:
    rs=[r for r in tests if r['order']==order and r['domain']==domain and r['method']==method]
    dm=defaultdict(list)
    for r in rs:
     for d in r['documents']:dm[d['hash']].append(d)
    maps[method]={h:{'nll':st.mean(d['nll'] for d in ds),'targets':ds[0]['targets'],'group':ds[0]['group']} for h,ds in dm.items()}
    assert all(len(v)==3 for v in dm.values())
   assert set(maps[a])==set(maps[b])
   groups=defaultdict(lambda:[0.,0.,0])
   for h,d in maps[a].items():
    e=maps[b][h];assert d['targets']==e['targets'] and d['group']==e['group']
    g=groups[d['group']];g[0]+=d['nll'];g[1]+=e['nll'];g[2]+=d['targets']
   values=np.array(list(groups.values()));G=len(values)
   rng=np.random.default_rng(20261010)
   idx=rng.integers(0,G,size=(2000,G));totals=values[idx].sum(axis=1)
   diffs=(totals[:,0]-totals[:,1])/totals[:,2]
   paired.append({'order':order,'domain':domain,'comparison':a+' minus '+b,'difference_nats':float((values[:,0].sum()-values[:,1].sum())/values[:,2].sum()),'ci95_low':float(np.quantile(diffs,.025)),'ci95_high':float(np.quantile(diffs,.975)),'groups':G,'documents':len(maps[a]),'bootstrap_draws':2000,'interpretation':'descriptive; grouped term resampling conditional on fixed models, core and split; negative favors MCE'})
order_checks=[]
for method in METHODS:
 for domain in ['general','civil','penal']:
  a=[r for r in tests if r['method']==method and r['domain']==domain and r['order']==ORDERS[0]]
  b=[r for r in tests if r['method']==method and r['domain']==domain and r['order']==ORDERS[1]]
  delta=[next(x for x in b if x['seed']==s)['loss_nats']-next(x for x in a if x['seed']==s)['loss_nats'] for s in SEEDS]
  order_checks.append({'method':method,'domain':domain,'reverse_minus_forward':ms(delta),'maximum_absolute_paired_difference':max(map(abs,delta))})
report={'run':done['run'],'summary':summaries,'paired_bootstrap':paired,'order_comparisons':order_checks,'completion':done,'baseline_test':{d:{k:v for k,v in r.items() if k not in ('documents','windows')} for d,r in baseline.items()},'audit':{'general_train_documents':audit['general_training_documents_scanned'],'flagged_by_domain_and_split':{}}}
counts=defaultdict(int)
for r in audit['records']:
 if r['flagged']:counts[r['domain']+'_'+r['split']]+=1
report['audit']['flagged_by_domain_and_split']=dict(counts)
(out/'analyse_extension_mce.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
flat=[]
for r in summaries:
 row={'ordre':' → '.join(r['order']),'methode':r['method'],'lr':r['lr'],'params':r['trainable_parameters_per_domain']}
 for domain in ['general','civil','penal']:
  for stat in ['mean','sd']:row[domain+'_test_'+stat]=r[domain+'_test'][stat]
 row['oubli_premier_domaine_validation']=r['first_domain_forgetting_validation']['mean'];row['temps_s']=r['training_seconds_two_domains']['mean'];row['gpu_GB']=r['peak_gpu_GB'];flat.append(row)
with (out/'tableau_test_extension.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
with (out/'intervalles_apparies_extension.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axes=plt.subplots(2,3,figsize=(11.5,7.3))
for ri,order in enumerate(ORDERS):
 for ci,domain in enumerate(['general','civil','penal']):
  ax=axes[ri,ci];rs=[r for r in summaries if r['order']==order]
  ax.bar(range(5),[r[domain+'_test']['mean'] for r in rs],yerr=[r[domain+'_test']['sd'] for r in rs],color=[COLORS[r['method']] for r in rs],capsize=3)
  ax.axhline(baseline[domain]['loss_nats'],color='#555',ls='--',lw=1,label='Cœur avant adaptation')
  ax.set_xticks(range(5),[LABEL[m] for m in METHODS],rotation=25,ha='right')
  ax.set_title(('Civil → pénal' if ri==0 else 'Pénal → civil')+' | '+domain)
  ax.set_ylabel('Test : nats / token');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(fontsize=8)
fig.suptitle('Test réservé après réglage sur validation — moyenne ± écart-type de 3 graines',fontsize=12)
fig.text(.5,.012,'Un cœur, un glossaire, deux ordres. Test juridique par document ; général sur 8192 cibles fixes.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(out/'figure_test_extension.png',dpi=200);fig.savefig(out/'figure_test_extension.svg');plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,6.8))
for ri,order in enumerate(ORDERS):
 for ci,domain in enumerate(['civil','penal']):
  ax=axes[ri,ci]
  for method in METHODS:
   points=[(0,0)]+[(stage,step) for stage in [1,2] for step in [100,200,300]]
   values=[]
   for stage,step in points:
    a=[r['loss_nats'] for r in curves if r['phase']=='evaluation' and r['order']==order and r['method']==method and r['domain']==domain and r['stage']==stage and r['step']==step]
    assert len(a)==3;values.append(ms(a))
   ax.errorbar([0,100,200,300,400,500,600],[r['mean'] for r in values],yerr=[r['sd'] for r in values],color=COLORS[method],label=LABEL[method],marker='o',markersize=3,capsize=2)
  ax.axvline(300,color='#777',ls='--',lw=.8);ax.set_title(('Civil → pénal' if ri==0 else 'Pénal → civil')+' | '+domain)
  ax.set_xlabel('Pas cumulés des deux domaines');ax.set_ylabel('Validation : nats / token');ax.grid(alpha=.2)
handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',ncol=5,bbox_to_anchor=(.5,.035))
fig.text(.5,.012,'Courbes descriptives ; les checkpoints sont fixés au pas 300 par domaine, sans sélection du meilleur point.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(out/'figure_courbes_extension.png',dpi=200);fig.savefig(out/'figure_courbes_extension.svg');plt.close(fig)

tables=[]
for order in ORDERS:
 tables.append('### '+' → '.join(order)+'\n\n| Méthode | LR retenu | Test général | Test civil | Test pénal | Oubli du premier domaine (validation) |\n|---|---:|---:|---:|---:|---:|')
 for r in summaries:
  if r['order']==order:tables.append(f"| {LABEL[r['method']]} | {r['lr']:.4f} | {fmt(r['general_test'])} | {fmt(r['civil_test'])} | {fmt(r['penal_test'])} | {fmt(r['first_domain_forgetting_validation'])} |")
 tables.append('')
(out/'tables_extension.md').write_text('\n'.join(tables)+'\n')
print(json.dumps({'completion':done,'summary':flat,'paired':paired,'overlap_flags':dict(counts)},ensure_ascii=False,indent=2))
