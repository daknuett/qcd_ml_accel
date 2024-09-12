import torch
import qcd_ml_accel

from itertools import product


def pool(features, L_coarse, block_size):
    res = torch.zeros(*L_coarse, *features.shape[-2:], dtype=torch.cdouble)
    for x,y,z,t in product(*tuple([range(block_size[i]) for i in range(4)])):
        res += features[x::block_size[0], y::block_size[1], z::block_size[2], t::block_size[3]]
    return res


def test_v_pool4d():
    fine_v = torch.randn(8, 8, 8, 8, 4, 3, dtype=torch.cdouble)
    L_coarse = [2, 2, 2, 2]
    block_size = [4, 4, 4, 4]
    coarse_v = pool(fine_v, L_coarse, block_size)

    pooled = qcd_ml_accel.pool4d.v_pool4d(fine_v, torch.tensor(block_size, dtype=torch.uint64));

    assert torch.allclose(coarse_v, pooled)
