"""The Triton activation kernel (beta fused) must produce exactly the bytes the fake-quant
reference produces: packed FP4 data, E4M3 scales (after un-swizzling) and the per-tensor scale,
for both NVFP4 modes and for decode/prefill-like shapes, including Llama-like outliers.
Also checks the swizzled layout against torchao's to_blocked, and the GEMM against
a dequantized FP32 reference."""
import _paths
import torch
import nvfp4_numerics as N
import nvfp4_kernels as K
from torchao.prototype.mx_formats.utils import to_blocked

torch.manual_seed(0)
dev = "cuda"
ok_all = True
for two in (True, False):
    for M, Kd in [(1, 4096), (16, 4096), (82, 14336), (1312, 4096), (300, 1024)]:
        x = (torch.randn(M, Kd, device=dev) * 2).bfloat16()
        x[:, 7] *= 300                                   # a massive-activation channel
        col = torch.rand(Kd, device=dev) * 50 + 0.5
        q, s, g = K.quantize_act(x, col, two_level=two)
        xs = x.float() * col
        gref = N.tensor_scale(xs.abs().amax()) if two else None
        codes, s8, _ = N.nvfp4_quantize_ref(xs, global_scale=gref)
        ok_q = torch.equal(q, N.pack_fp4(codes))
        nrb, ncb = -(-M // 128), -(-(Kd // 16) // 4)
        s_un = s.view(nrb, ncb, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(nrb * 128, ncb * 4)[:M, :Kd // 16]
        ok_s = torch.equal(s_un, s8.view(torch.uint8))
        # layout: where does torchao's to_blocked put element (r, c)? compare at those slots
        ids = torch.arange(1, M * (Kd // 16) + 1, device=dev, dtype=torch.int32).view(M, Kd // 16)
        where = to_blocked(ids)
        valid = where > 0
        ok_layout = torch.equal(s[valid], s8.view(torch.uint8).flatten()[where[valid].long() - 1])
        ok_g = (not two) or torch.equal(g.reshape(1), gref.reshape(1))
        ok = ok_q and ok_s and ok_g and ok_layout
        ok_all &= ok
        print(f"{'two' if two else 'one'}-level M={M:<5} K={Kd:<6} packed={ok_q} scales={ok_s} "
              f"tensor_scale={ok_g} layout={ok_layout}")

# GEMM vs dequantized reference
for M in (1, 16, 82, 1312):
    Kd, Nd = 4096, 1024
    w = torch.randn(Nd, Kd, device=dev) * 0.02
    x = (torch.randn(M, Kd, device=dev)).bfloat16()
    col = torch.rand(Kd, device=dev) + 0.5
    lin = K.NVFP4W4A4Linear(w.bfloat16(), col)
    y = lin(x).float()
    q, s, g = K.quantize_act(x, col)
    xs = x.float() * col
    xd = N.nvfp4_fake_quant(xs)
    yref = xd @ lin.dequant_weight().t()
    rel = ((y - yref).norm() / yref.norm()).item()
    ok = rel < 5e-3
    ok_all &= ok
    print(f"GEMM M={M:<5} rel err vs dequantized fp32 reference {rel:.2e}  {'OK' if ok else 'FAIL'}")
print("ALL OK" if ok_all else "FAILURES")
