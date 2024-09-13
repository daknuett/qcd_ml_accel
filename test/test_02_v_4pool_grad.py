import torch
import qcd_ml_accel
import pytest


@pytest.mark.slow
def test_v_pool4d_grad():
    fine_v = torch.randn(8, 8, 8, 8, 4, 3, dtype=torch.cdouble, requires_grad=True)
    L_coarse = [2, 2, 2, 2]
    block_size = [4, 4, 4, 4]

    assert torch.autograd.gradcheck(qcd_ml_accel.pool4d.v_pool4d, (fine_v, torch.tensor(block_size, dtype=torch.int64)))


def test_v_pool4d_grad_small():
    fine_v = torch.randn(2, 2, 2, 2, 4, 3, dtype=torch.cdouble, requires_grad=True)
    L_coarse = [2, 2, 2, 2]
    block_size = [2, 2, 2, 2]

    assert torch.autograd.gradcheck(qcd_ml_accel.pool4d.v_pool4d, (fine_v, torch.tensor(block_size, dtype=torch.int64)))

@pytest.mark.slow
def test_v_unpool4d_grad():
    coarse_v = torch.randn(2, 2, 2, 2, 4, 3, dtype=torch.cdouble, requires_grad=True)
    L_coarse = [2, 2, 2, 2]
    block_size = [4, 4, 4, 4]

    assert torch.autograd.gradcheck(qcd_ml_accel.pool4d.v_unpool4d, (coarse_v, torch.tensor(block_size, dtype=torch.int64)))

def test_v_unpool4d_grad_small():
    coarse_v = torch.randn(2, 2, 2, 2, 4, 3, dtype=torch.cdouble, requires_grad=True)
    L_coarse = [2, 2, 2, 2]
    block_size = [2, 2, 2, 2]

    assert torch.autograd.gradcheck(qcd_ml_accel.pool4d.v_unpool4d, (coarse_v, torch.tensor(block_size, dtype=torch.int64)))
