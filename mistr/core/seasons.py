"""Month -> season mapping of the composited time steps."""
import torch


def season_index(lr_start: torch.Tensor, lr_end: torch.Tensor, out_start: torch.Tensor,
                 out_end: torch.Tensor) -> torch.Tensor:
    """(B, T) index of the lr step whose [start, end] contains each output step's midpoint;
    -1 where none does."""
    mid = (out_start + out_end).double()[:, :, None] / 2                     # (B, T, 1)
    inside = (lr_start[:, None, :] <= mid) & (mid <= lr_end[:, None, :])     # (B, T, G)
    return torch.where(inside.any(-1), inside.double().argmax(-1),
                       torch.full_like(out_start, -1)).long()


def seasons_used(season: torch.Tensor, present: torch.Tensor, n_seasons: int) -> torch.Tensor:
    """(B, n_seasons) bool: the season holds at least one present step. (Not a scatter_ of
    `present`: a padded step writes False over the real steps of the same season.)"""
    real = present & (season >= 0)
    hot = torch.nn.functional.one_hot(season.clamp(min=0), n_seasons).bool()   # (B, T, S)
    return (hot & real[..., None]).any(1)
