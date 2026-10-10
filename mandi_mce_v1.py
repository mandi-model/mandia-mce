"""Experimental MCE V1 and per-domain LoRA bank. Routing is explicit; no learned router.
Requires mandi_model.py. Use eval() for inference. Only the newest domain may train.
Single synchronous forward only; concurrent LoRA calls are not supported.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F
from mandi_model import MandiLM


def _valid_name(name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',name):
        raise ValueError('Use a simple domain identifier, e.g. law or history')


def _loss(logits,targets):
    return None if targets is None else F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))


class ResidualExpert(nn.Module):
    def __init__(self,width,bottleneck=256,dropout=.1):
        super().__init__()
        self.norm=nn.RMSNorm(width,eps=1e-5)
        self.down=nn.Linear(width,bottleneck,bias=False)
        self.up=nn.Linear(bottleneck,width,bias=False)
        self.dropout=nn.Dropout(dropout)
        nn.init.normal_(self.down.weight,std=.02)
        nn.init.zeros_(self.up.weight)

    def forward(self,h):
        return h+self.up(self.dropout(F.gelu(self.down(self.norm(h)))))


class MandiMCE(nn.Module):
    """One residual adapter per core block and domain; prior modules remain frozen."""
    def __init__(self,core,bottleneck=256,dropout=.1):
        super().__init__()
        self.core=core.requires_grad_(False)
        self.core.zero_grad(set_to_none=True)
        self.experts=nn.ModuleDict()
        self.bottleneck=bottleneck
        self.dropout=dropout
        self.active_train_domain=None
        self.core.eval()

    def add_expert(self,name):
        _valid_name(name)
        if name in self.experts: raise ValueError('Domain already exists: '+name)
        for p in self.experts.parameters():
            p.requires_grad_(False); p.grad=None
        device=next(self.core.parameters()).device
        dtype=next(self.core.parameters()).dtype
        self.experts[name]=nn.ModuleList([ResidualExpert(self.core.cfg['width'],self.bottleneck,self.dropout)
                                          for _ in self.core.blocks]).to(device=device,dtype=dtype)
        self.active_train_domain=name
        self.train(self.training)
        return list(self.experts[name].parameters())

    def train(self,mode=True):
        super().train(mode)
        self.core.eval()
        for name,expert in self.experts.items(): expert.train(mode and name==self.active_train_domain)
        return self

    def forward(self,ids,targets=None,expert=None):
        if self.training and torch.is_grad_enabled() and expert!=self.active_train_domain:
            raise ValueError("Training must select the newest domain; use eval() for inference")
        if expert is None: return self.core(ids,targets)
        if expert not in self.experts: raise KeyError(expert)
        if ids.shape[1]>self.core.cfg['context']: raise ValueError('Context exceeded')
        h=self.core.token_embedding(ids)
        for block,adapter in zip(self.core.blocks,self.experts[expert]): h=adapter(block(h))
        logits=self.core.lm_head(self.core.final_norm(h))
        return logits,_loss(logits,targets)

    def domain_parameters(self,name):
        return list(self.experts[name].parameters())

    def frozen_core_parameters(self):
        return list(self.core.named_parameters())

    def named_domain_parameters(self,name):
        return list(self.experts[name].named_parameters())


class LowRankUpdate(nn.Module):
    def __init__(self,inputs,outputs,rank,alpha,dropout):
        super().__init__()
        self.a=nn.Linear(inputs,rank,bias=False)
        self.b=nn.Linear(rank,outputs,bias=False)
        self.dropout=nn.Dropout(dropout)
        self.scale=alpha/rank
        nn.init.kaiming_uniform_(self.a.weight,a=5**.5)
        nn.init.zeros_(self.b.weight)

    def forward(self,x):
        return self.b(self.a(self.dropout(x)))*self.scale


class BankLinear(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.adapters=nn.ModuleDict()
        self.active=None

    def forward(self,x):
        out=self.base(x)
        return out if self.active is None else out+self.adapters[self.active](x)


class MandiLoRABank(nn.Module):
    """Separate unmerged LoRA weights for each explicit domain; all original weights frozen."""
    def __init__(self,core,rank=8,alpha=None,dropout=.1):
        super().__init__()
        if not isinstance(rank,int) or rank<=0: raise ValueError('Positive rank required')
        self.core=core.requires_grad_(False)
        self.core.zero_grad(set_to_none=True)
        self.rank=rank
        self.alpha=2*rank if alpha is None else alpha
        self.dropout=dropout
        self.domains=[]
        self.active_train_domain=None
        for block in self.core.blocks:
            block.attention.qkv=BankLinear(block.attention.qkv)
            block.attention.projection=BankLinear(block.attention.projection)
            for key in ['gate','up','down']:
                setattr(block.feed_forward,key,BankLinear(getattr(block.feed_forward,key)))
        self.core.eval()

    def _linears(self):
        return [m for m in self.core.modules() if isinstance(m,BankLinear)]

    def add_expert(self,name):
        _valid_name(name)
        if name in self.domains: raise ValueError('Domain already exists: '+name)
        for linear in self._linears():
            for p in linear.adapters.parameters():
                p.requires_grad_(False); p.grad=None
            w=linear.base.weight
            linear.adapters[name]=LowRankUpdate(linear.base.in_features,linear.base.out_features,
                                                self.rank,self.alpha,self.dropout).to(device=w.device,dtype=w.dtype)
        self.domains.append(name)
        self.active_train_domain=name
        self.train(self.training)
        return self.domain_parameters(name)

    def train(self,mode=True):
        super().train(mode)
        self.core.eval()
        for linear in self._linears():
            for name,adapter in linear.adapters.items(): adapter.train(mode and name==self.active_train_domain)
        return self

    def forward(self,ids,targets=None,expert=None):
        if expert is not None and expert not in self.domains: raise KeyError(expert)
        if self.training and torch.is_grad_enabled() and expert!=self.active_train_domain:
            raise ValueError("Training must select the newest domain; use eval() for inference")
        linears=self._linears()
        previous=[linear.active for linear in linears]
        try:
            for linear in linears: linear.active=expert
            return self.core(ids,targets)
        finally:
            for linear,active in zip(linears,previous): linear.active=active

    def domain_parameters(self,name):
        if name not in self.domains: raise KeyError(name)
        return [p for linear in self._linears() for p in linear.adapters[name].parameters()]

    def frozen_core_parameters(self):
        return [(n,p) for n,p in self.core.named_parameters() if '.adapters.' not in n]

    def named_domain_parameters(self,name):
        marker='.adapters.'+name+'.'
        return [(n,p) for n,p in self.core.named_parameters() if marker in n]
