# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

from contextlib import nullcontext
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16
defaultcfg = {
    # 6 : [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    # 6 : [[510, 375, 512, 443], [512, 399, 479, 286], [511, 367, 370, 196], [512, 404, 111, 95], [512, 425, 60, 66], [512, 365, 356, 223]]
    6 : [[360, 512], [408, 479], [360, 370], [408, 111], [432, 60], [360, 356]]
}

class channel_selection2(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection2, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))
        self.grads = nn.Parameter(torch.zeros(num_channels), requires_grad=False)
        self.hessian_diagonal = nn.Parameter(torch.zeros(num_channels, num_channels), requires_grad=False)

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, hidden_feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats=feats, hidden_feats=hidden_feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp1 = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        out = self.msa(self.la1(x), mask) + x
        out = self.mlp2(self.mlp1(self.la2(out))) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, hidden_feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.hidden_feats = hidden_feats
        self.sqrt_d = self.hidden_feats**0.5

        self.q = nn.Linear(feats, hidden_feats, bias=False)
        self.k = nn.Linear(feats, hidden_feats, bias=False)
        self.v = nn.Linear(feats, hidden_feats, bias=False)

        self.o = nn.Linear(hidden_feats, feats)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.hidden_feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.hidden_feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.hidden_feats//self.head).transpose(1,2)



        dots = torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d #(b,h,n,n)
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        score = F.softmax(dots, dim=-1)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class Transformer(nn.Module):
    def __init__(self, hidden, mlp_hidden, dropout, head, num_layers, cfg):
        super().__init__()
        self.layers = nn.ModuleList([])
        if cfg is not None:
            for num in cfg:
                self.layers.append(
                  TransformerEncoder(feats=hidden, hidden_feats=num[0], mlp_hidden=num[1], dropout=dropout, head=head)
                )
        else:
            for _ in range(num_layers):
                self.layers.append(
                  TransformerEncoder(feats=hidden, hidden_feats=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head)
                )
        
    def forward(self, x, mask=None):
        for encoder in self.layers:
            x = encoder(x, mask)
        return x

class ViT11_slim(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., cfg=None, num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT11_slim, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        self.transformer = Transformer(hidden=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head, cfg=cfg, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x, mask=None):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.transformer(out, mask)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out
