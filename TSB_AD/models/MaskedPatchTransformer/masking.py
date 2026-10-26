"""Complementary group masking: g(c, n) = (n // block + c + rot) mod G, staggered across channels."""

import torch


def group_ids(n_channels, n_patches, groups, block=1, rotation=0, device=None):
    """Group index of every (channel, patch) token → LongTensor [C, N]."""
    c = torch.arange(n_channels, device=device).unsqueeze(1)      # [C, 1]
    n = torch.arange(n_patches, device=device).unsqueeze(0)       # [1, N]
    return ((n // max(1, int(block))) + c + int(rotation)) % int(groups)


def mask_for_group(n_channels, n_patches, groups, target, block=1, rotation=0,
                   device=None):
    """Boolean [C, N]: True where the token is masked (belongs to `target`)."""
    return group_ids(n_channels, n_patches, groups, block, rotation,
                     device) == int(target)


def coverage_is_complete(n_channels, n_patches, groups, block=1, rotation=0,
                         device=None):
    """True if every token is masked exactly once across the G passes."""
    gid = group_ids(n_channels, n_patches, groups, block, rotation, device)
    counts = torch.zeros(n_channels, n_patches, dtype=torch.long, device=device)
    for g in range(int(groups)):
        counts += (gid == g).long()
    return bool((counts == 1).all())
