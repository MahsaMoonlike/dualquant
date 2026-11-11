"""Batched GPU BCD (dq_nvfp4_bcd.dualquant_nvfp4) vs the repo's own per-block CPU loop
(layer_wrapper_dualquant.wrap_dualquant_layer) on a real Llama-3.1-8B projection.
Both use the same NVFP4 cast; they differ only in device and summation order."""
import time
import _paths
import torch
import torch.nn as nn
from layer_wrapper_dualquant import wrap_dualquant_layer
from dq_nvfp4_bcd import dualquant_nvfp4

name = "model.layers.0.self_attn.k_proj.weight"
W = _paths.load_weight(name)
cfg = dict(num_splits=1, scale_option="row_column", col_init="l1_norm", row_init="max_abs",
           block_size=16, num_iter=15, scale_format="e4m3", dualquant_to_element_tensor=False)

lin = nn.Linear(W.shape[1], W.shape[0], bias=False)
lin.weight.data.copy_(W.float())
t0 = time.time()
col_ref = wrap_dualquant_layer(lin, None, cfg, "nvfp4", act_quant_flag=True)
t_ref = time.time() - t0
mq_ref = lin.weight.data.clone()

t0 = time.time()
mq, col = dualquant_nvfp4(W)
torch.cuda.synchronize()
t_new = time.time() - t0
mq, col = mq.cpu(), col.cpu()

rel = lambda a, b: ((a - b).norm() / b.norm()).item()
Wf = W.float()
mse = lambda m, c: ((Wf - m * c[None]) ** 2).mean().item()
print(f"{name}  shape={tuple(W.shape)}")
print(f"  reference CPU loop  : {t_ref:7.2f} s   recon MSE {mse(mq_ref, col_ref):.4e}")
print(f"  batched GPU         : {t_new:7.2f} s   recon MSE {mse(mq, col):.4e}")
print(f"  mat_q  rel diff {rel(mq, mq_ref):.2e}   identical elems {(mq == mq_ref).float().mean().item()*100:.3f}%")
print(f"  beta   rel diff {rel(col, col_ref):.2e}")
print(f"  RTN NVFP4 recon MSE (no scales): {((Wf - __import__('nvfp4_numerics').nvfp4_fake_quant(Wf)) ** 2).mean().item():.4e}")
