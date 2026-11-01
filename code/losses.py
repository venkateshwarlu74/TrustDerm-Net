"""Losses: evidential (Eq. 15), hierarchical class-balanced (Eq. 16), confidence-weighted
distillation (Eqs. 17-19) and concept BCE (Eq. 11)."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from model import alpha


def class_balanced_weights(counts, beta=0.999):
    """Effective-number weights [63], normalised to sum to the number of classes."""
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
    w = (1 - beta) / (1 - np.power(beta, counts))
    return torch.tensor(w / w.sum() * len(w), dtype=torch.float32)


def edl(a, target, k, lam):
    """Per-sample evidential loss (Eq. 15) for Dirichlet parameters a (B x k)."""
    y = F.one_hot(target, k).float()
    S = a.sum(1, keepdim=True)
    nll = (y * (torch.digamma(S) - torch.digamma(a))).sum(1)
    at = y + (1 - y) * a
    St = at.sum(1, keepdim=True)
    kl = (torch.lgamma(St).squeeze(1) - math.lgamma(k) - torch.lgamma(at).sum(1)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St))).sum(1))
    return nll + lam * kl


def _head_loss(logits, target, k, lam, evidential):
    if evidential:
        return edl(alpha(logits), target, k, lam)
    return F.cross_entropy(logits.float(), target, reduction="none")


def hierarchical_loss(out, y, z, w7, w2, lam, hierarchical=True, evidential=True):
    """Eq. 16: Stage-1 loss for every sample + Stage-2 loss only on the true branch."""
    if not hierarchical:
        return (w7[y] * _head_loss(out["s7"], y, 7, lam, evidential)).mean()
    l1 = w2[z] * _head_loss(out["s1"], z, 2, lam, evidential)
    l2 = torch.zeros_like(l1)
    b, m = z == 0, z == 1
    if b.any():
        l2[b] = w7[y[b]] * _head_loss(out["sb"][b], y[b], 4, lam, evidential)
    if m.any():
        l2[m] = w7[y[m]] * _head_loss(out["sm"][m], y[m] - 4, 3, lam, evidential)
    return (l1 + l2).mean()


def _student_logits(s, evidential):
    return torch.log(alpha(s)) if evidential else s.float()        # softmax(log a) = Dirichlet mean


def kd_loss(s_out, t_out, tau, weighted, hierarchical=True, evidential=True):
    """Eqs. 17-19. The teacher is always hierarchical with softmax heads."""
    with torch.no_grad():
        p1, pb, pm = (F.softmax(t_out[h].float(), 1) for h in ("s1", "sb", "sm"))
        q7 = torch.cat([p1[:, :1] * pb, p1[:, 1:] * pm], 1).clamp_min(1e-8)
        w = 1 - (-(q7 * q7.log()).sum(1)) / math.log(7) if weighted else torch.ones(q7.shape[0], device=q7.device)
    kl = torch.zeros_like(w)
    if hierarchical:
        for h in ("s1", "sb", "sm"):
            q = F.softmax(t_out[h].float() / tau, 1)
            logp = F.log_softmax(_student_logits(s_out[h], evidential) / tau, 1)
            kl = kl + F.kl_div(logp, q, reduction="none").sum(1)
    else:
        q = q7.pow(1 / tau)
        q = q / q.sum(1, keepdim=True)
        logp = F.log_softmax(_student_logits(s_out["s7"], evidential) / tau, 1)
        kl = F.kl_div(logp, q, reduction="none").sum(1)
    return tau ** 2 * (w * kl).mean()


def concept_loss(out, concepts):
    if "concept_logits" not in out:
        return None
    mask = concepts[:, 0] >= 0
    if not mask.any():
        return None
    return F.binary_cross_entropy_with_logits(out["concept_logits"][mask].float(), concepts[mask])
