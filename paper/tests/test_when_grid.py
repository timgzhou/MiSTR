"""select.when=grid: one step per window, cycling over the sample's used coarse steps."""
import torch

from mistr_paper.core.selection import grid_steps


def test_grid_steps_cycle_over_used_steps():
    used = torch.tensor([[True, False, True, True],      # 3 used steps: 0, 2, 3
                         [False, True, False, False],    # 1 used step
                         [False, False, False, False]])  # none
    g = grid_steps(used, 4)                              # (B, NS, G, G)
    assert g.shape == (3, 4, 4, 4)
    assert (g[:2].sum(1) == 1).all() and not g[2].any()  # exactly one step per window
    assert not g[0, 1].any()                             # unused step never picked
    steps = g[0].float().argmax(0)                       # (G, G) chosen step index
    want = torch.tensor([0, 2, 3])[(torch.arange(4)[:, None] + torch.arange(4)[None]) % 3]
    assert torch.equal(steps, want)
    assert g[1, 1].all()
    # balanced: every used step gets ~1/n of the windows
    counts = g[0].sum((1, 2))[used[0]]
    assert counts.max() - counts.min() <= 1
