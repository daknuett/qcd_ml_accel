import torch
from torch import Tensor

__all__ = ["v_pool4d"]

def v_pool4d(v_spincolor_field: Tensor, block_size: Tensor) -> Tensor:
    """
    Performs a 4D pooling operation on a 4D spin-color field.

    I.e. for a 4D spin-color field of shape (Lx, Ly, Lz, Lt, Nc, Ns),
    and a block size of (bsx, bsy, bsz, bst), the output will be a 4D 
    spin-color field of shape (Lx//bsx, Ly//bsy, Lz//bsz, Lt//bst, Nc, Ns),
    where each element of the output is the sum of the elements in the
    corresponding block of the input.
    """
    return torch.ops.qcd_ml_accel.v_pool4d.default(v_spincolor_field, block_size)

@torch.library.register_fake("qcd_ml_accel::v_pool4d")
def _(v_spincolor_field, block_size):
    torch._check(v_spincolor_field.dtype == torch.cdouble)
    torch._check(block_size.dtype == torch.uint64)
    L_coarse = [lfi // bsi for lfi, bsi in zip(v_spincolor_field.shape[:-2], block_size)]
    return torch.empty(*L_coarse, *v_spincolor_field.shape[-2:], dtype=torch.cdouble)


