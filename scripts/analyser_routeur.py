"""Recompute router accuracy, clustered intervals and routed losses from observations."""
from pathlib import Path
from collections import defaultdict
import json, statistics, sys
import numpy as np

root=Path(__file__).resolve().parents[1]
raw=root/'results/evidence'
pred=json.loads((raw/'router_test_predictions.json').read_text())
metrics=json.loads((raw/'router_metrics.json').read_text())
routed=json.loads((raw/'routed_test_results.json').read_text())
labels=['civil','penal']
assert len(pred)==155 and len(routed)==48
def measures(rows):
    matrix=np.zeros((2,2),dtype=int)
    for r in rows:matrix[labels.index(r['true_domain']),labels.index(r['route'])]+=1
    f1=[]
    for i in range(2):
        denom=int(matrix[i,:].sum()+matrix[:,i].sum())
        f1.append(2*int(matrix[i,i])/denom if denom else 0.)
    return float(np.trace(matrix)/len(rows)),statistics.mean(f1),matrix.tolist()
acc,f1,matrix=measures(pred)
assert abs(acc-metrics['test_accuracy'])<1e-12
assert abs(f1-metrics['test_macro_f1'])<1e-12
assert matrix==metrics['confusion_matrix_civil_penal']
groups=defaultdict(list)
for r in pred:groups[r['group']].append(r)
g=list(groups.values());rng=np.random.default_rng(20261010);values=[]
for _ in range(2000):
    sample=[r for i in rng.integers(0,len(g),len(g)) for r in g[i]]
    values.append(measures(sample)[:2])
ci=np.quantile(values,[.025,.975],axis=0)
assert np.allclose(ci[:,0],metrics['accuracy_ci95'],atol=1e-12,rtol=0)
assert np.allclose(ci[:,1],metrics['macro_f1_ci95'],atol=1e-12,rtol=0)
tables=[]
for order in [('civil','penal'),('penal','civil')]:
    for method in ['MCE','MCE64','LoRA_r8','LoRA_r32']:
        for domain in labels:
            rows=[r for r in routed if tuple(r['order'])==order and r['method']==method and r['domain']==domain]
            assert len(rows)==3
            for r in rows:
                assert sum(d['targets'] for d in r['documents'])==r['targets']
                assert abs(sum(d['nll'] for d in r['documents'])/r['targets']-r['routed_loss_nats'])<1e-10
            tables.append({'order':order,'method':method,'domain':domain,
                           **{k:statistics.mean(r[k] for r in rows) for k in
                              ['routed_loss_nats','oracle_loss_nats','routing_penalty_nats']}})
report={'accuracy':acc,'macro_f1':f1,'confusion_matrix':matrix,'test_terms':len(pred),
        'term_groups':len(g),'accuracy_ci95':ci[:,0].tolist(),'macro_f1_ci95':ci[:,1].tolist(),
        'scope':'Closed-set supplied legal terms; teacher-forced definition/EOS completion. Conditional descriptive group bootstrap.',
        'routed_loss_means_three_seeds':tables}
dest=Path(sys.argv[1]) if len(sys.argv)>1 else root/'analysis-output/router_reanalysis.json'
dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Router metrics, group intervals and 48 routed results verified:',dest)
