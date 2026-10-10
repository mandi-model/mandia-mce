import math
import torch
from torch import nn
from torch.nn import functional as F

class MandiAttention(nn.Module):
    def __init__(self, width, heads, context, dropout):
        super().__init__()
        assert width % heads == 0 and (width//heads) % 2 == 0
        self.heads, self.head_dim, self.dropout = heads, width//heads, dropout
        self.qkv = nn.Linear(width, 3*width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0,self.head_dim,2).float()/self.head_dim))
        angles = torch.arange(context).float()[:,None]*inv_freq[None,:]
        self.register_buffer('rope_cos', angles.cos(), persistent=False)
        self.register_buffer('rope_sin', angles.sin(), persistent=False)
    def rotate(self, x):
        t = x.shape[-2]
        cos = self.rope_cos[:t].to(dtype=x.dtype)[None,None,:,:]
        sin = self.rope_sin[:t].to(dtype=x.dtype)[None,None,:,:]
        a,b = x.chunk(2, dim=-1)
        return torch.cat((a*cos-b*sin,b*cos+a*sin), dim=-1)
    def forward(self, x):
        b,t,w = x.shape
        q,k,v = self.qkv(x).view(b,t,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        q,k = self.rotate(q), self.rotate(k)
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True,
             dropout_p=self.dropout if self.training else 0.0)
        return self.projection(y.transpose(1,2).contiguous().view(b,t,w))

class MandiFeedForward(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width,hidden,bias=False)
        self.up = nn.Linear(width,hidden,bias=False)
        self.down = nn.Linear(hidden,width,bias=False)
    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))

class MandiBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attention_norm = nn.RMSNorm(cfg['width'],eps=1e-5)
        self.attention = MandiAttention(cfg['width'],cfg['heads'],cfg['context'],cfg['dropout'])
        self.feed_forward_norm = nn.RMSNorm(cfg['width'],eps=1e-5)
        self.feed_forward = MandiFeedForward(cfg['width'],cfg['ffn'])
        self.dropout = nn.Dropout(cfg['dropout'])
    def forward(self,x):
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.feed_forward(self.feed_forward_norm(x)))

class MandiLM(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.token_embedding = nn.Embedding(cfg['vocab_size'],cfg['width'])
        self.blocks = nn.ModuleList([MandiBlock(cfg) for _ in range(cfg['layers'])])
        self.final_norm = nn.RMSNorm(cfg['width'],eps=1e-5)
        self.lm_head = nn.Linear(cfg['width'],cfg['vocab_size'],bias=False)
        self.apply(self.init_weights)
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight,std=0.02/math.sqrt(2*cfg['layers']))
            nn.init.normal_(block.feed_forward.down.weight,std=0.02/math.sqrt(2*cfg['layers']))
        self.lm_head.weight = self.token_embedding.weight
    @staticmethod
    def init_weights(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
    def forward(self,ids,targets=None):
        if ids.shape[1]>self.cfg['context']: raise ValueError('Contexte dépassé')
        x = self.token_embedding(ids)
        for block in self.blocks: x=block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
        return logits,loss

def mandi_optimizer(model,lr):
    decay=[p for p in model.parameters() if p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.ndim<2]
    return torch.optim.AdamW([{'params':decay,'weight_decay':0.1},
                             {'params':no_decay,'weight_decay':0.0}],lr=lr,betas=(0.9,0.95))
