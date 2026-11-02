"""A CUDA reference for checking mpsplat (gsplat on Apple GPUs) against gsplat on CUDA, for pySLAM.

pySLAM's Gaussian splatting (MonoGS) will call `gsplat.rasterization(..., render_mode="RGB+D")` and refine
keyframe poses by autograd through the view matrix: viewmat = SE3_exp(tau) @ T, as MonoGS's
utils/pose_utils.py does. This script makes fixed scenes, renders them, and takes the gradients of a fixed
loss with respect to every input, including tau and the 2D means (which densification uses).

    python pyslam/reference.py make --out pyslam/reference_cuda.npz      # on CUDA (done on an RTX 5090)
    python pyslam/reference.py compare --ref pyslam/reference_cuda.npz --device mps
    python pyslam/reference.py fd --device mps                          # finite-difference check, no reference
    python pyslam/reference.py speed --device mps                       # fwd+bwd ms at 640x480, 100k and 300k
"""
import argparse
import time

import numpy as np
import torch

from gsplat import rasterization

# the scenes: (name, number of Gaussians, width, height)
kScenes = [("small", 50, 64, 48), ("medium", 5000, 320, 240)]
kInputs = ["means", "quats", "log_scales", "opacity_logits", "colors", "tau"]


def skew(x):
    z = torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.stack([torch.stack([z, -x[2], x[1]]), torch.stack([x[2], z, -x[0]]), torch.stack([-x[1], x[0], z])])


def se3_exp(tau):
    """MonoGS's SE3_exp (rho = tau[:3], theta = tau[3:]), written without in-place writes."""
    rho, theta = tau[:3], tau[3:]
    W = skew(theta)
    W2 = W @ W
    I = torch.eye(3, dtype=tau.dtype, device=tau.device)
    angle = torch.norm(theta)
    if angle < 1e-5:
        R = I + W + 0.5 * W2
        Vm = I + 0.5 * W + W2 / 6.0
    else:
        R = I + (torch.sin(angle) / angle) * W + ((1 - torch.cos(angle)) / angle**2) * W2
        Vm = I + W * ((1 - torch.cos(angle)) / angle**2) + W2 * ((angle - torch.sin(angle)) / angle**3)
    t = Vm @ rho
    top = torch.cat([R, t[:, None]], dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=tau.dtype, device=tau.device)
    return torch.cat([top, bottom], dim=0)


def make_scene(n, width, height, seed):
    """Gaussians 2-4 m in front of a camera at a fixed pose, MonoGS's parametrisation, and the loss weights."""
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g) - 0.5) * torch.tensor([1.0, 0.75])  # within the central 80% of the view
    z = 2.0 + 2.0 * torch.rand(n, 1, generator=g)
    s = {
        "means": torch.cat([xy * z, z], dim=1),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "log_scales": torch.log(0.01 + 0.05 * torch.rand(n, 3, generator=g)),
        "opacity_logits": torch.randn(n, generator=g),
        "colors": torch.rand(n, 3, generator=g),
        "tau": 1e-3 * torch.randn(6, generator=g),  # a small pose update, as in MonoGS's pose window
        "T": torch.eye(4),
        "K": torch.tensor([[0.8 * width, 0, width / 2], [0, 0.8 * width, height / 2], [0, 0, 1.0]]),
        "w_rgb": torch.randn(height, width, 3, generator=g),
        "w_depth": torch.randn(height, width, 1, generator=g),
        "w_alpha": torch.randn(height, width, 1, generator=g),
    }
    a = 0.3
    s["T"][:3, :3] = torch.tensor([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=torch.float32)
    s["T"][:3, 3] = torch.tensor([0.1, -0.05, 0.2])
    # the means were drawn in the camera's frame (2-4 m ahead, inside the view): put them in the world frame,
    # so that no Gaussian is at the edge of the view or near the camera, where the loss is not smooth
    R, t = s["T"][:3, :3], s["T"][:3, 3]
    s["means"] = (s["means"] - t) @ R
    return s


def render_and_grad(s, width, height, device):
    """Render (RGB, accumulated depth as MonoGS renders it, alpha) and the gradients of a fixed linear loss."""
    p = {k: s[k].to(device).clone().requires_grad_(True) for k in kInputs}
    viewmat = (se3_exp(p["tau"]) @ s["T"].to(device))[None]
    renders, alphas, info = rasterization(
        p["means"], p["quats"], p["log_scales"].exp(), torch.sigmoid(p["opacity_logits"]), p["colors"],
        viewmat, s["K"].to(device)[None], width, height, render_mode="RGB+D", packed=False,
    )
    info["means2d"].retain_grad()
    rgb, depth, alpha = renders[0, ..., :3], renders[0, ..., 3:4], alphas[0]
    loss = (rgb * s["w_rgb"].to(device)).sum() + (depth * s["w_depth"].to(device)).sum() + (alpha * s["w_alpha"].to(device)).sum()
    loss.backward()
    out = {"rgb": rgb, "depth": depth, "alpha": alpha, "loss": loss, "radii": info["radii"][0],
           "grad_means2d": info["means2d"].grad[0]}
    out.update({f"grad_{k}": p[k].grad for k in kInputs})
    return {k: v.detach().float().cpu().numpy() for k, v in out.items()}


def make(args):
    data = {}
    for seed, (name, n, w, h) in enumerate(kScenes):
        s = make_scene(n, w, h, seed=seed)
        for k, v in s.items():
            data[f"{name}/in/{k}"] = v.numpy()
        for k, v in render_and_grad(s, w, h, "cuda").items():
            data[f"{name}/out/{k}"] = v
    data["meta/torch"] = np.array(torch.__version__)
    data["meta/device"] = np.array(torch.cuda.get_device_name(0))
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}")


def compare(args):
    ref = np.load(args.ref)
    print(f"reference: torch {ref['meta/torch']}, {ref['meta/device']}; this: torch {torch.__version__}, {args.device}")
    for name, n, w, h in kScenes:
        s = {k.split("/")[-1]: torch.from_numpy(ref[k]) for k in ref.files if k.startswith(f"{name}/in/")}
        got = render_and_grad(s, w, h, args.device)
        print(f"-- {name}: {n} Gaussians, {w}x{h}")
        for k, v in got.items():
            r = ref[f"{name}/out/{k}"]
            if k == "radii":
                print(f"  {k:22s} differ for {int((v != r).reshape(n, -1).any(-1).sum())} of {n} Gaussians "
                      f"({int((r.reshape(n, -1) > 0).any(-1).sum())} visible in the reference)")
                continue
            err = np.abs(v - r).max()
            rel = err / max(np.abs(r).max(), 1e-12)
            print(f"  {k:22s} max abs diff {err:.3e}   relative to max |ref| {rel:.3e}")


def fd(args):
    """Central finite differences of the loss for a few inputs, in float32 (expect ~1e-2 relative agreement)."""
    name, n, w, h = kScenes[0]
    s = make_scene(n, w, h, seed=0)
    base = render_and_grad(s, w, h, args.device)
    eps = 1e-3
    for k in ["means", "log_scales", "opacity_logits", "colors", "tau"]:
        flat = s[k].reshape(-1)
        idx = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(1))[:5]
        for i in idx.tolist():
            plus, minus = dict(s), dict(s)
            plus[k] = flat.clone(); plus[k][i] += eps; plus[k] = plus[k].reshape(s[k].shape)
            minus[k] = flat.clone(); minus[k][i] -= eps; minus[k] = minus[k].reshape(s[k].shape)
            num = (render_and_grad(plus, w, h, args.device)["loss"] - render_and_grad(minus, w, h, args.device)["loss"]) / (2 * eps)
            ana = base[f"grad_{k}"].reshape(-1)[i]
            print(f"  d loss / d {k}[{i}]: analytic {ana:+.4e}  numeric {float(num):+.4e}")


def speed(args):
    for n in (100_000, 300_000):
        s = make_scene(n, 640, 480, seed=2)
        times = []
        for it in range(12):
            if args.device == "cuda":
                torch.cuda.synchronize()
            elif args.device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            render_and_grad(s, 640, 480, args.device)  # includes the copy of the outputs to the CPU
            times.append(time.perf_counter() - t0)
        print(f"  {n} Gaussians, 640x480: forward+backward median {1000 * np.median(times[2:]):.1f} ms (10 runs after 2 warm-ups)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["make", "compare", "fd", "speed"])
    ap.add_argument("--out", default="pyslam/reference_cuda.npz")
    ap.add_argument("--ref", default="pyslam/reference_cuda.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"make": make, "compare": compare, "fd": fd, "speed": speed}[args.command](args)
