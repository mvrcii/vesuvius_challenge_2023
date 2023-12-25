import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


# helper function
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def createViT3D():
    image_size = (48, 48)
    layers = 12
    num_classes = 1
    channels = 1

    # working example
    image_patch_size = (8, 8)  # Size of patches each frame is divided into (height, width)
    frame_patch_size = 3  # Number of frames grouped into a single temporal patch
    dim = 512  # Dimensionality of token embeddings in the transformer
    depth = 6  # Number of layers (blocks) in the transformer
    heads = 8  # Number of attention heads in each transformer layer
    mlp_dim = 1024  # Dimensionality of the feedforward network in each transformer layer
    dim_head = 64  # Dimensionality of each attention head
    dropout = 0.1  # Dropout rate used in attention and feedforward networks
    emb_dropout = 0.1  # Dropout rate for token

    # big
    # image_patch_size = (4, 4)  # Smaller patches for finer details
    # frame_patch_size = 2  # More frames per patch for better temporal resolution
    # dim = 768  # Increased embedding dimension
    # depth = 12  # More layers
    # heads = 12  # More attention heads
    # mlp_dim = 3072  # Larger feedforward network
    # dim_head = 64  # Increased head dimension
    # dropout = 0.2  # Standard dropout
    # emb_dropout = 0.2  # Standard embedding dropout

    # huge
    # image_patch_size = (4, 4)  # Smaller patches for high resolution
    # frame_patch_size = 2  # High temporal resolution
    # dim = 1024  # Very large embedding dimension
    # depth = 24  # Many layers for deep feature extraction
    # heads = 16  # Many attention heads
    # mlp_dim = 4096  # Very large feedforward network
    # channels = 3  # Standard for RGB video frames
    # dim_head = 64  # Balancing head dimension with total dimension
    # dropout = 0.3  # Higher dropout to counteract overfitting
    # emb_dropout = 0.3  # Higher embedding dropout for same reas

    return ViT3D(
        image_size=image_size,
        image_patch_size=image_patch_size,
        frames=layers,
        frame_patch_size=frame_patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        pool='cls',  # Pooling method ('cls' for class token, 'mean' for mean pooling)
        channels=channels,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout
    )


class ViT3D(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads,
                 mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
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

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
