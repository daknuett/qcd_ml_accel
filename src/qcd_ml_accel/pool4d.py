import torch
from torch import Tensor

__all__ = ["v_pool4d", "v_unpool4d"]

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


def v_unpool4d(v_spincolor_field: Tensor, block_size: Tensor) -> Tensor:
    """
    Performs a 4D unpooling operation on a 4D spin-color field.

    I.e. for a 4D spin-color field of shape (Lx, Ly, Lz, Lt, Nc, Ns),
    and a block size of (bsx, bsy, bsz, bst), the output will be a 4D 
    spin-color field of shape (Lx*bsx, Ly*bsy, Lz*bsz, Lt*bst, Nc, Ns),
    where each block of the output is the corresponding element of the input.
    """
    return torch.ops.qcd_ml_accel.v_unpool4d.default(v_spincolor_field, block_size)

@torch.library.register_fake("qcd_ml_accel::v_unpool4d")
def _(v_spincolor_field, block_size):
    torch._check(v_spincolor_field.dtype == torch.cdouble)
    torch._check(block_size.dtype == torch.uint64)
    L_fine = [lci * bsi for lci, bsi in zip(v_spincolor_field.shape[:-2], block_size)]
    return torch.empty(*L_fine, *v_spincolor_field.shape[-2:], dtype=torch.cdouble)


def _backward_v_pool4d(ctx, grad):
    v_spincolor_field, block_size = ctx.saved_tensors
    grad_v_spincolor_field = None
    if ctx.needs_input_grad[0]:
        grad_v_spincolor_field = torch.ops.qcd_ml_accel.v_unpool4d.default(grad, block_size)
    return grad_v_spincolor_field, None

def _setup_context_v_pool4d(ctx, inputs, output):
    v_spincolor_field, block_size = inputs
    ctx.save_for_backward(v_spincolor_field, block_size)

torch.library.register_autograd("qcd_ml_accel::v_pool4d", _backward_v_pool4d, setup_context=_setup_context_v_pool4d)


def _backward_v_unpool4d(ctx, grad):
    v_spincolor_field, block_size = ctx.saved_tensors
    grad_v_spincolor_field = None
    if ctx.needs_input_grad[0]:
        grad_v_spincolor_field = torch.ops.qcd_ml_accel.v_pool4d.default(grad, block_size)
    return grad_v_spincolor_field, None

def _setup_context_v_unpool4d(ctx, inputs, output):
    v_spincolor_field, block_size = inputs
    ctx.save_for_backward(v_spincolor_field, block_size)

torch.library.register_autograd("qcd_ml_accel::v_unpool4d", _backward_v_unpool4d, setup_context=_setup_context_v_unpool4d)
