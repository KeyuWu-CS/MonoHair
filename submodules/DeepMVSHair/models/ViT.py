import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # attention

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        # re-attention

        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)
        attn = self.reattn_norm(attn)

        # aggregate and out

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class DeepViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class OccViT(nn.Module):
    def __init__(self, *, output_dim, token_dim, feat_dim, pt_dim, depth, heads, mlp_dim, num_views,
                 pool ='cls', dim_head = 64, dropout = 0., emb_dropout = 0., use_pos=True, use_pt=True, fuse_func='vit'):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.num_views = num_views
        self.use_pos = use_pos
        self.use_pt = use_pt
        self.fuse_func = fuse_func
        print('==> in vit init')
        print('fuse func: ', self.fuse_func)
        print('use pos: ', self.use_pos)
        print('use pt: ', self.use_pt)
        self.view_fuse_pt = nn.Linear(feat_dim + pt_dim if use_pt else feat_dim, token_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.cls_fuse_pt = nn.Linear(token_dim + pt_dim if use_pt else token_dim, token_dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_views + 1, token_dim)) if use_pos else None

        self.transformer = Transformer(token_dim, depth, heads, dim_head, mlp_dim, dropout) if fuse_func == 'vit' else None
        self.mlp_fuse = nn.Linear(num_views * token_dim, token_dim) if fuse_func == 'mlp' else None

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, output_dim)
        )

    def forward(self, data):
        '''

        :param x: [N, V, C_ft], N: batch size (num of sample points); V: number of views; C1: feature channels
        :param cam_embed: [V, C_cam], V: number of views; C1: extrinsic parameters embedding
        :param pt_embed: [N, C_pt], N: batch size (num of sample points); C3: point coordinate embedding
        :return:
        '''
        # [N, V, C_ft]
        img_feat = data['img_feat']
        # [N, 1, C_pt]
        pts_world_feat = data['pts_world_feat'] if self.use_pt else None
        # [N, V, C_pt]
        pts_view_feat = data['pts_view_feat'] if self.use_pt else None

        n, v, _ = img_feat.shape

        # [N, V, C_ft] -> [N, V, C_tk]
        view_tokens = self.view_fuse_pt(torch.cat([img_feat, pts_view_feat], dim=-1) if self.use_pt else img_feat)
        # print('view tokens', view_tokens.shape)
        # directly use average pooling, for ablation study (w.o. ViT)
        if self.fuse_func == 'avg':
            avg_token = view_tokens.mean(dim=1)
            return self.mlp_head(avg_token)
        elif self.fuse_func == 'mlp':
            view_perm = torch.randperm(self.num_views)
            view_tokens = view_tokens[:, view_perm]
            view_tokens_flat = view_tokens.reshape(n, -1)
            fused_token = self.mlp_fuse(view_tokens_flat)
            return self.mlp_head(fused_token)
        else:
            # [1, 1, C_tk] -> [N, 1, C_tk]
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=n)
            cls_tokens = self.cls_fuse_pt(torch.cat([cls_tokens, pts_world_feat], dim=-1) if self.use_pt else cls_tokens)

            # [N, V + 1, C_tk]
            y = torch.cat((cls_tokens, view_tokens), dim=1)
            if self.use_pos:
                y += self.pos_embedding
            # print('final input', y.shape)

            y = self.dropout(y)

            y = self.transformer(y)

            y = y.mean(dim=1) if self.pool == 'mean' else y[:, 0]

            y = self.to_latent(y)
            return self.mlp_head(y)
