import numpy as np
import torch

SPEC = {"type": "read", "model": "original", "what": "queries", "n_images": 50, "top": 200}
A_GATE = 2.0    # weight on the primary-query (objectness) direction
BETA = 2.0      # weight on the species direction
T_BIRD = 10.0   # class logit (before species term) of an average primary bird query


def edit(train, weights):
    r = train.read(SPEC)
    E = np.asarray(r["embedding"], dtype=np.float64)
    II = np.asarray(r["image_index"])
    L = np.asarray(r["logits"])
    MS = np.asarray(r["mask_score"])
    MC = np.asarray(r["matched_class"])
    W0 = weights["class_rows.weight"].numpy()
    b0 = weights["class_rows.bias"].numpy()
    n_cls = W0.shape[0] - 1

    # primary query per image: the query matching the labelled bird with the lowest no-object logit
    prim, lab = [], []
    for i in np.unique(II):
        s = np.where((II == i) & (MC >= 0))[0]
        if len(s) == 0:
            continue
        j = s[np.argmin(L[s, n_cls])]
        prim.append(j); lab.append(MC[j] - 1)
    prim = np.array(prim); lab = np.array(lab)
    P = E[prim]
    g = P.mean(0)

    # gate direction: mean primary minus mean of every other confident-mask query (duplicates, parts, background)
    is_prim = np.zeros(len(E), bool); is_prim[prim] = True
    d = g - E[~is_prim & (MS > 0.5)].mean(0)
    d /= np.linalg.norm(d)
    t = float((P @ d).mean())

    # species directions: unit(class mean of primaries - mean of all primaries)
    N = np.zeros((n_cls, E.shape[1]))
    for c in range(n_cls):
        if (lab == c).any():
            v = P[lab == c].mean(0) - g
            N[c] = v / np.linalg.norm(v)

    Wn = W0.astype(np.float64).copy()
    bn = b0.astype(np.float64).copy()
    Wn[:n_cls] = A_GATE * d[None, :] + BETA * N
    bn[:n_cls] = -A_GATE * t + T_BIRD - BETA * (N @ g)
    # no-object row (last) left unchanged
    return {"class_rows.weight": torch.tensor(Wn, dtype=torch.float32),
            "class_rows.bias": torch.tensor(bn, dtype=torch.float32)}
