"""Restore the private civil/penal pilot checkpoints. This is not a legal assistant.
Requires mandi_model.py and mandi_mce_v1.py next to this file.
"""
import hashlib
from pathlib import Path
import torch
from mandi_model import MandiLM
from mandi_mce_v1 import MandiMCE, MandiLoRABank


def checkpoint_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def charger_pilote(core_path,expert_path,device='cpu',expected_expert_sha256=None):
    """Load exact frozen routes and return (model, metadata); inference mode by default."""
    core_path,expert_path=Path(core_path),Path(expert_path)
    if expected_expert_sha256 is not None and checkpoint_sha256(expert_path)!=expected_expert_sha256:
        raise ValueError('Expert checkpoint fingerprint does not match the manifest')
    expert=torch.load(expert_path,map_location='cpu',weights_only=True)
    if checkpoint_sha256(core_path)!=expert['core_checkpoint_sha256']:
        raise ValueError('Core checkpoint fingerprint does not match this expert bank')
    base=torch.load(core_path,map_location='cpu',weights_only=True)
    if base['config']!=expert['core_config']:raise ValueError('Core configurations differ')
    core=MandiLM(base['config']);core.load_state_dict(base['model'],strict=True)
    method=expert['method'];domains=expert['domains']
    if domains not in [['civil','penal'],['penal','civil']]:raise ValueError('Unexpected legal domains/order')
    if method=='full':
        model=core;model.load_state_dict(expert['state'],strict=True)
    elif method in ['MCE','MCE64','LoRA_r8','LoRA_r32']:
        model=MandiMCE(core,bottleneck=expert.get('bottleneck',256)) if method.startswith('MCE') else MandiLoRABank(core,rank=8 if method=='LoRA_r8' else 32)
        for domain in domains:model.add_expert(domain)
        expected={k for k in model.state_dict() if k.startswith('experts.')} if method.startswith('MCE') else {k for k in model.state_dict() if '.adapters.' in k}
        if set(expert['state'])!=expected:raise ValueError('Missing or unexpected adapter weights')
        status=model.load_state_dict(expert['state'],strict=False)
        if status.unexpected_keys or set(status.missing_keys)!=(set(model.state_dict())-expected):
            raise ValueError('Unexpected state loading result')
    else:raise ValueError('Unknown pilot method: '+str(method))
    model.to(device);model.requires_grad_(False);model.zero_grad(set_to_none=True);model.eval()
    metadata={k:expert[k] for k in ['method','seed','domains','core_checkpoint_sha256','scope']}
    metadata.update({k:expert[k] for k in ['source_license','lr','bottleneck','rank','order'] if k in expert})
    del base,expert
    return model,metadata


@torch.no_grad()
def prediction_pilote(model,metadata,token_ids,domaine=None):
    """Return logits. Modular routes: None=general, 'civil' or 'penal'."""
    model.eval()
    if domaine is not None and domaine not in metadata['domains']:raise ValueError('Unknown domain')
    return model(token_ids)[0] if metadata['method']=='full' else model(token_ids,expert=domaine)[0]
