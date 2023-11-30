# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16

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
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
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
        self.select2 = channel_selection2(mlp_hidden)

    def forward(self, x, mask=None):
        out = self.msa(self.la1(x), mask) + x
        out = self.mlp2(self.select2(self.mlp1(self.la2(out)))) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats, bias=False)
        self.k = nn.Linear(feats, feats, bias=False)
        self.v = nn.Linear(feats, feats, bias=False)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

        self.select1 = channel_selection2(feats)

    def forward(self, x, mask=None):
        b, n, f = x.size()
        q = (self.select1(self.q(x))).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = (self.select1(self.k(x))).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = (self.select1(self.v(x))).view(b, n, self.head, self.feats//self.head).transpose(1,2)



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
    def __init__(self, hidden, mlp_hidden, dropout, head, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head)
            )
    def forward(self, x, mask=None):
        for encoder in self.layers:
            x = encoder(x, mask)
        return x

class ViT11(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT11, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        self.transformer = Transformer(hidden=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head, num_layers=num_layers)
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

if __name__ == "__main__":
    # setup_seed(200)
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = ViT11(
    in_c=3,
    img_size = 32,
    patch = 4,
    num_classes = 10,
    hidden = 384,                  # 512
    num_layers = 7,
    head = 8,
    mlp_hidden = 384*4,
    dropout = 0.1,
    is_cls_token=True
    )
    y = net(x)
    # print(y)
    print(y.size())
    for k,v in net.state_dict().items():
        print(k)
        print("\n")
    # with torch.no_grad():
    #     for k, m in enumerate(net.modules()):
    #         if k==11:
    #             print(m)
    #         if k==13:
    #             print(m)
    #         if isinstance(m, channel_selection2):
    #             print(k)
    #             print(m)
            # print(m.indexes.data)