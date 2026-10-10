"""Render English-language figures from the published numerical observations."""
from pathlib import Path
import csv,json,statistics as st,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]) if len(sys.argv)>1 else root/'results/figures'
out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((root/'results/tableau_test_extension.csv').open()))
curves=json.loads((root/'results/evidence/validation_curves.json').read_text())
baseline=json.loads((root/'results/evidence/test_baseline.json').read_text())
methods=['full','MCE','MCE64','LoRA_r8','LoRA_r32']
labels={'full':'Full','MCE':'MCE 256','MCE64':'MCE 64','LoRA_r8':'LoRA r8','LoRA_r32':'LoRA r32'}
colors={'full':'#b44a3a','MCE':'#126e76','MCE64':'#599e91','LoRA_r8':'#7c6ca7','LoRA_r32':'#c49321'}
orders=[['civil','penal'],['penal','civil']]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axes=plt.subplots(2,3,figsize=(11.5,7.3))
for ri,order in enumerate(orders):
    selected=[r for r in rows if r['ordre']==' → '.join(order)]
    assert [r['methode'] for r in selected]==methods
    for ci,domain in enumerate(['general','civil','penal']):
        ax=axes[ri,ci]
        ax.bar(range(5),[float(r[domain+'_test_mean']) for r in selected],
               yerr=[float(r[domain+'_test_sd']) for r in selected],
               color=[colors[m] for m in methods],capsize=3)
        ax.axhline(baseline[domain]['loss_nats'],color='#555',ls='--',lw=1,label='Core before adaptation')
        ax.set_xticks(range(5),[labels[m] for m in methods],rotation=25,ha='right')
        ax.set_title(' → '.join(d.title() for d in order)+' | '+domain.title())
        ax.set_ylabel('Test loss: nats / token');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(fontsize=8)
fig.suptitle('Held-out test after validation-based tuning: mean ± SD across 3 seeds',fontsize=12)
fig.text(.5,.012,'One core, one glossary, two orders. Legal test by document; general test on 8,192 fixed targets.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(out/'test_results.en.png',dpi=200);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,6.8))
for ri,order in enumerate(orders):
    for ci,domain in enumerate(['civil','penal']):
        ax=axes[ri,ci]
        for method in methods:
            values=[]
            for stage,step in [(0,0)]+[(stage,step) for stage in [1,2] for step in [100,200,300]]:
                v=[r['loss_nats'] for r in curves if r['phase']=='evaluation' and r['order']==order
                   and r['method']==method and r['domain']==domain and r['stage']==stage and r['step']==step]
                assert len(v)==3;values.append((st.mean(v),st.stdev(v)))
            ax.errorbar([0,100,200,300,400,500,600],[v[0] for v in values],yerr=[v[1] for v in values],
                        color=colors[method],label=labels[method],marker='o',markersize=3,capsize=2)
        ax.axvline(300,color='#777',ls='--',lw=.8)
        ax.set_title(' → '.join(d.title() for d in order)+' | '+domain.title())
        ax.set_xlabel('Cumulative updates across both domains');ax.set_ylabel('Validation loss: nats / token');ax.grid(alpha=.2)
handles,names=axes[0,0].get_legend_handles_labels()
fig.legend(handles,names,loc='lower center',ncol=5,bbox_to_anchor=(.5,.035))
fig.text(.5,.012,'Descriptive curves; checkpoints fixed at update 300 per domain, without best-point selection.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(out/'validation_curves.en.png',dpi=200);plt.close(fig)
print('English figures rendered from unchanged observations:',out)
