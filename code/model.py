"""TrustDerm-Net: image encoder + metadata tokens + gated cross-attention fusion (Eqs. 4-9),
concept bottleneck (Eq. 10) and hierarchical evidential heads (Eqs. 12-14).
Every component can be switched off for the ablation study."""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONCEPTS, SITES


def cat(xs):
    """Concatenate in float32: under autocast, Linear outputs fp16 while LayerNorm/Embedding output fp32."""
    return torch.cat([t.float() for t in xs], 1)


class TrustDermNet(nn.Module):
    def __init__(self, backbone="mobilevit_s", img_size=224, d=256, n_heads=4, d_r=16, pretrained=True,
                 use_meta=True, fusion="gated", hierarchical=True, evidential=True, use_concepts=True,
                 p_md=0.3, p_drop_all=0.1):
        super().__init__()
        self.cfg = dict(backbone=backbone, img_size=img_size, d=d, n_heads=n_heads, d_r=d_r, use_meta=use_meta,
                        fusion=fusion, hierarchical=hierarchical, evidential=evidential,
                        use_concepts=use_concepts, p_md=p_md, p_drop_all=p_drop_all)
        kw = dict(pretrained=pretrained, num_classes=0, global_pool="")
        if "vit_" in backbone and "mobilevit" not in backbone:
            kw["img_size"] = img_size
        self.encoder = timm.create_model(backbone, **kw)
        self.n_prefix = getattr(self.encoder, "num_prefix_tokens", 0)
        self.proj = nn.Linear(self.encoder.num_features, d)
        self.use_meta, self.fusion, self.p_md, self.p_drop_all = use_meta, fusion, p_md, p_drop_all
        self.hierarchical, self.evidential, self.use_concepts = hierarchical, evidential, use_concepts

        if use_meta:
            self.age = nn.Linear(1, d)
            self.sex = nn.Embedding(2, d)
            self.site = nn.Embedding(len(SITES), d)
            self.miss = nn.Parameter(torch.randn(3, d) * 0.02)    # one learned 'missing' token per field
            if fusion == "gated":
                self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
                self.gate = nn.Linear(2 * d, d)
                self.out = nn.Linear(d, d)
            else:                                                  # ablation: plain concatenation
                self.concat = nn.Linear(4 * d, d)
        self.norm = nn.LayerNorm(d)

        n_c = len(CONCEPTS)
        if use_concepts:
            self.concept = nn.Linear(d, n_c)
            self.resid = nn.Linear(d, d_r)
        s1_in = n_c + d_r if use_concepts else d
        if hierarchical:
            self.head1 = nn.Linear(s1_in, 2)
            self.head_b = nn.Linear(d, 4)
            self.head_m = nn.Linear(d, 3)
        else:                                                      # ablation: flat 7-class head
            self.head7 = nn.Linear(s1_in + d if use_concepts else d, 7)

    def tokens(self, x):
        f = self.encoder.forward_features(x)
        if f.ndim == 4:      # CNN maps are B x C x H x W; Swin-style maps are channels-last B x H x W x C
            f = f.flatten(1, 2) if f.shape[-1] == self.encoder.num_features else f.flatten(2).transpose(1, 2)
        else:
            f = f[:, self.n_prefix:]
        return self.proj(f)                                        # B x L x d   (Eq. 4)

    def meta_tokens(self, age, sex, site):
        B = age.shape[0]
        a = torch.nan_to_num(age, nan=0.0).unsqueeze(1) / 100.0
        E = torch.stack([self.age(a).float(), self.sex(sex.clamp(min=0)).float(),
                         self.site(site.clamp(min=0)).float()], 1)                                 # Eq. 5
        missing = torch.stack([torch.isnan(age), sex < 0, site < 0], 1)
        if self.training and self.p_md > 0:                        # modality dropout (Eq. 6)
            missing = missing | (torch.rand(B, 3, device=age.device) < self.p_md)
            missing = missing | (torch.rand(B, 1, device=age.device) < self.p_drop_all)
        return torch.where(missing.unsqueeze(-1), self.miss.unsqueeze(0).expand(B, -1, -1).float(), E)

    def forward(self, x, age, sex, site):
        T_ = self.tokens(x)
        g = T_.mean(1)
        if not self.use_meta:
            v = self.norm(g)
        else:
            E = self.meta_tokens(age, sex, site)
            if self.fusion == "gated":
                O, _ = self.attn(E.to(T_.dtype), T_, T_)            # Eqs. 7-8
                gamma = torch.sigmoid(self.gate(cat([g, E.mean(1)])))
                v = self.norm(g.float() + gamma.float() * self.out(O.mean(1)).float())  # Eq. 9
            else:
                v = self.norm(self.concat(cat([g, E.flatten(1)])))
        out = {}
        if self.use_concepts:
            out["concept_logits"] = self.concept(v)                 # Eq. 10
            s1_in = cat([torch.sigmoid(out["concept_logits"]), self.resid(v)])
        else:
            s1_in = v
        if self.hierarchical:
            out["s1"], out["sb"], out["sm"] = self.head1(s1_in), self.head_b(v), self.head_m(v)
        else:
            out["s7"] = self.head7(cat([s1_in, v]) if self.use_concepts else v)
        return out


def alpha(logits):
    return F.softplus(logits.float()) + 1.0                         # Eq. 12


def probs_and_uncertainty(out, hierarchical, evidential):
    """Composite 7-class probabilities (Eq. 13), P(malignant) and uncertainty u (Eq. 14)."""
    def head(s, k):
        if evidential:
            a = alpha(s)
            S = a.sum(1, keepdim=True)
            return a / S, (k / S).squeeze(1)
        p = F.softmax(s.float(), 1)
        return p, 1.0 - p.max(1).values

    if hierarchical:
        p1, u1 = head(out["s1"], 2)
        pb, ub = head(out["sb"], 4)
        pm, um = head(out["sm"], 3)
        p7 = torch.cat([p1[:, :1] * pb, p1[:, 1:] * pm], 1)
        u = torch.maximum(u1, torch.where(p1[:, 1] >= 0.5, um, ub))
        return p7, p1[:, 1], u
    p7, u = head(out["s7"], 7)
    return p7, p7[:, 4:].sum(1), u


def build(cfg, pretrained=True):
    return TrustDermNet(pretrained=pretrained, **cfg)
