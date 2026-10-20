"""Complexity / speed benchmark for GNet (network-only and full-pipeline).

Reports parameters, GMACs/GFLOPs (via thop), per-image latency, throughput and
peak GPU memory for a given configuration. FLOPs are for the forward pass of the
two-input network (current frame + background).
"""
import argparse
import time
import torch

from models.gnet import GNet


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=None)
    p.add_argument("--fusion", default="cat", choices=["cat", "sub"])
    p.add_argument("--tsam-depth", type=int, default=12)
    p.add_argument("--size", type=int, default=513)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--label", default="")
    p.add_argument("--no-flops", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = get_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    m = GNet(num_classes=1, tsam_depth=a.tsam_depth, fusion=a.fusion,
             pretrained=False).to(dev).eval()
    if a.weights:
        ck = torch.load(a.weights, map_location="cpu")
        m.load_state_dict(ck["model"])
    nparam = sum(p.numel() for p in m.parameters())

    gmacs = gflops = float("nan")
    if not a.no_flops and dev.type == "cuda":
        try:
            from thop import profile
            xf = torch.randn(1, 3, a.size, a.size, device=dev)
            bf = torch.randn(1, 3, a.size, a.size, device=dev)
            macs, _ = profile(m, inputs=(xf, bf), verbose=False)
            gmacs = macs / 1e9
            gflops = 2 * macs / 1e9
        except Exception as e:
            print(f"[flops] failed: {e}")

    x = torch.randn(a.batch, 3, a.size, a.size, device=dev)
    b = torch.randn(a.batch, 3, a.size, a.size, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(5):
            m(x, b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(a.iters):
            m(x, b)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t) / a.iters
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0
    tag = f"[{a.label}] " if a.label else ""
    print(f"{tag}params(M)={nparam/1e6:.2f} fusion={a.fusion} depth={a.tsam_depth} "
          f"batch={a.batch} size={a.size} GMACs={gmacs:.1f} GFLOPs={gflops:.1f} "
          f"latency={dt*1000:.1f}ms fps={a.batch/dt:.1f} peakGB={peak:.2f}")


if __name__ == "__main__":
    main()
