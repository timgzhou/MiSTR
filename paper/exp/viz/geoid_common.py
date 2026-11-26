"""Shared GEOID-Flood tracking figure code: build and run the round-O EventTrack models on one test
series (fixed S_L, MiSTR S_L & S_C, fixed S_C), the 3-class colour map, mIoU and flood boundary IoU."""
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from exp.geoidchange.tiletrack import EventTrack, biou_counts, collate  # noqa: E402

def fbiou(p, y):
    """Flood-class boundary IoU on one map (the round-N headline bIoU)."""
    i, u = biou_counts(torch.as_tensor(p)[None], torch.as_tensor(y)[None], 3)
    return float(i[2] / u[2]) if u[2] > 0 else float("nan")

CK = "checkpoints/geoidchange"
Q_UNIT = "abs"                         # round O: q_unit=series (q = share of all the series' window slots)
plt.rcParams.update({"font.family": ["Liberation Serif", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 8})
CMAP = ListedColormap(["#d4d4d0", "#f7f7f5", "#5cc8b8", "#2a78d6"])               # -1, dry, permanent, flood
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 8})

def miou(p, y):
    ok = y >= 0
    out = []
    for c in range(3):
        u = (((p == c) | (y == c)) & ok).sum()
        if u:
            out.append(((p == c) & (y == c) & ok).sum() / u)
    return float(np.mean(out))


def models(T, R, seed, q, combo="64c32", prefix=None, suffix=""):
    """fixed S_L, MiSTR S_L & S_C, fixed S_C (combo 'LcC'), checkpoints <prefix><name>_s<seed>.pt; suffix '_qr' =
    the budget-range MiSTR (round N method)."""
    b = prefix or f"trkL_T{T}_r{R}_"
    L, C = (int(v) for v in combo.split("c"))
    return [("fixL", f"fixed $S_{{{L}}}$", f"{b}fix{L}_s{seed}", L, C, 0.0),
            ("mix", f"MiSTR $S_{{{L}}}$ & $S_{{{C}}}$ (q {q * 100:.4g} %)", f"{b}{combo}{suffix}_s{seed}", L, C, q),
            ("fixC", f"fixed $S_{{{C}}}$", f"{b}fix{C}_s{seed}", C, C, 0.0)]


BATCH = 1                              # > 1: the series shares a batch-wide crop budget with BATCH - 1 random test
                                       # series (= evaluation / training, budget=batch); only series 0 is drawn


@torch.no_grad()
def run(ms, ds, i, R):
    others = [int(j) for j in np.random.default_rng(i).choice([j for j in range(len(ds)) if j != i], BATCH - 1,
                                                             replace=False)] if BATCH > 1 else []
    s, d, v, post, y, _, rd, _ = collate([ds[i]] + [ds[j] for j in others])
    out = {}
    for key, lab, ck, lr, cp, q in ms:
        net = EventTrack(lr, cp, "self" if R == 0 else "cat", 3).cuda().eval()
        net.load_state_dict(torch.load(f"{CK}/{ck}.pt", map_location="cuda"))
        net.q_unit, net.ref_n = Q_UNIT, R
        net.budget_scope = "batch" if BATCH > 1 else "sample"
        o, _, info = net(s, d, v, post, "joint", q)
        boxes = None
        if q > 0:
            b0 = info["sel_b"].cpu().numpy() == 0                         # series 0's crops only
            boxes = {k: info[k].cpu().numpy()[b0] for k in ("sel_t", "win_y", "win_x")}
        crop0 = int((info["crop_b"] == 0).sum()) if q > 0 and info.get("crop_b") is not None else \
            (info["crops"] if q > 0 else 0)
        toks = int(v[0].sum()) * (512 // lr) ** 2 + crop0
        out[key] = (o.argmax(2)[0].cpu().numpy(), boxes, toks, cp)
        del net
    return out, int(post[0]), y[0].numpy(), rd[0].numpy(), v[0].numpy()
