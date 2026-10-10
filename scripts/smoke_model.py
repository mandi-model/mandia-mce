"""CPU checks with real updates on a miniature random model; no study data."""
from pathlib import Path
import sys,copy,io
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mandi_model import MandiLM
from mandi_mce_v1 import MandiMCE,MandiLoRABank
torch.set_num_threads(1)
cfg=dict(vocab_size=64,width=32,heads=4,context=16,dropout=.1,layers=2,ffn=64)
torch.manual_seed(7)
x=torch.randint(0,64,(2,12));y=torch.randint(0,64,(2,12))
for cls,kwargs in [(MandiMCE,dict(bottleneck=8)),(MandiMCE,dict(bottleneck=4)),(MandiLoRABank,dict(rank=2))]:
    model=cls(MandiLM(cfg),**kwargs)
    model.add_expert('civil'); model.train()
    opt=torch.optim.AdamW(model.domain_parameters('civil'),lr=.01)
    loss=model(x,y,expert='civil')[1]; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        old=model(x,expert='civil')[0].clone();general=model(x,expert=None)[0].clone()
    frozen={n:p.detach().clone() for n,p in model.named_parameters()}
    model.add_expert('penal');model.train()
    opt=torch.optim.AdamW(model.domain_parameters('penal'),lr=.01)
    before=[p.detach().clone() for p in model.domain_parameters('penal')]
    loss=model(x,y,expert='penal')[1];loss.backward()
    assert all(p.grad is None for _,p in model.frozen_core_parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.domain_parameters('civil'))
    opt.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,model.domain_parameters('penal')))
    for n,p in model.named_parameters():
        if n in frozen: assert torch.equal(frozen[n],p),n
    model.eval()
    with torch.no_grad():
        assert torch.equal(old,model(x,expert='civil')[0])
        assert torch.equal(general,model(x,expert=None)[0])
    restored=cls(MandiLM(cfg),**kwargs);restored.add_expert('civil');restored.add_expert('penal')
    stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
    restored.load_state_dict(torch.load(stream,weights_only=True));restored.eval()
    with torch.no_grad():
        assert torch.equal(model(x,expert='penal')[0],restored(x,expert='penal')[0])
    print(cls.__name__,kwargs,'update / gradients / frozen routes / reload: OK')
print('No legal capability or GPU experiment was measured by this check.')
