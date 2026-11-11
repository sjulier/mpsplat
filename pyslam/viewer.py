"""A live window of mpsplat (gsplat on Apple GPUs) rendering a small scene, and tracking the camera.

Top row: the scene rendered at the true camera, which orbits it (RGB, depth, alpha), with
render_mode="RGB+D" as pySLAM uses it.
Bottom row: a tracker that sees only the images. Each frame it predicts the pose with a constant
velocity model, then takes Adam steps on the pose update tau (view = SE3_exp(tau) @ T), as MonoGS's
tracking does. It shows its render, the error, and its pose error over time.

    python pyslam/viewer.py        # from the repository root; q closes the window, space pauses
"""
import math
import os
import sys
import time

import matplotlib

matplotlib.use("macosx")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

kHere = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(kHere))  # this repository's gsplat
sys.path.insert(0, kHere)
from gsplat import rasterization  # noqa: E402
from reference import se3_exp  # noqa: E402  MonoGS's SE3_exp, as in the CUDA reference

kDevice = "mps"
kWidth, kHeight = 320, 240
kTrackSteps = 20
kTrackLr = 5e-3
kStepDeg = 1.0  # orbit step per frame


def make_scene(g):
    """Gaussians on a checkered ground plane, a sphere coloured by its normal, and a cube (world y up)."""
    parts = []
    # ground plane y = 0, flat Gaussians
    n = 14000
    xz = (torch.rand(n, 2, generator=g) - 0.5) * 6.0
    means = torch.stack([xz[:, 0], torch.zeros(n), xz[:, 1]], 1)
    checker = ((torch.floor(xz[:, 0] / 0.75) + torch.floor(xz[:, 1] / 0.75)) % 2)[:, None]
    colors = checker * torch.tensor([0.85, 0.85, 0.8]) + (1 - checker) * torch.tensor([0.25, 0.3, 0.4])
    scales = torch.tensor([0.035, 0.004, 0.035]).repeat(n, 1)
    parts.append((means, colors, scales))
    # sphere, centre (0, 1, 0), radius 0.8
    n = 10000
    d = torch.nn.functional.normalize(torch.randn(n, 3, generator=g), dim=1)
    means = torch.tensor([0.0, 1.0, 0.0]) + 0.8 * d
    colors = 0.5 + 0.5 * d
    scales = torch.full((n, 3), 0.03)
    parts.append((means, colors, scales))
    # cube, centre (1.8, 0.45, 0.9), side 0.9: flat Gaussians on each face
    n_face = 1000
    face_colors = torch.tensor([[0.9, 0.2, 0.2], [0.2, 0.8, 0.3], [0.2, 0.4, 0.9],
                                [0.9, 0.8, 0.2], [0.8, 0.3, 0.8], [0.2, 0.8, 0.8]])
    for axis in range(3):
        for side, sign in enumerate((-1.0, 1.0)):
            uv = (torch.rand(n_face, 2, generator=g) - 0.5) * 0.9
            p = torch.zeros(n_face, 3)
            others = [a for a in range(3) if a != axis]
            p[:, others[0]], p[:, others[1]] = uv[:, 0], uv[:, 1]
            p[:, axis] = sign * 0.45
            s = torch.full((n_face, 3), 0.03)
            s[:, axis] = 0.004
            parts.append((p + torch.tensor([1.8, 0.45, 0.9]), face_colors[2 * axis + side].repeat(n_face, 1), s))
    means, colors, scales = (torch.cat([p[i] for p in parts]) for i in range(3))
    n = means.shape[0]
    return {
        "means": means.to(kDevice),
        "quats": torch.tensor([1.0, 0, 0, 0]).repeat(n, 1).to(kDevice),
        "scales": scales.to(kDevice),
        "opacities": torch.full((n,), 0.9).to(kDevice),
        "colors": colors.clamp(0, 1).to(kDevice),
    }


def look_at(centre, target):
    """World-to-camera matrix, OpenCV camera axes (x right, y down, z forward), world y up."""
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ centre
    return torch.tensor(T, dtype=torch.float32, device=kDevice)


def orbit_pose(deg):
    a = math.radians(deg)
    return look_at([4.5 * math.sin(a), 1.8, -4.5 * math.cos(a)], [0.5, 0.6, 0.3])


def render(scene, viewmat, K):
    out, alpha, _ = rasterization(scene["means"], scene["quats"], scene["scales"], scene["opacities"],
                                  scene["colors"], viewmat[None], K[None], kWidth, kHeight,
                                  render_mode="RGB+D", packed=False)
    return out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0]


def pose_error(T_est, T_true):
    """Camera-centre error (m) and rotation error (deg)."""
    D = (T_est @ torch.linalg.inv(T_true)).cpu().double()
    c_est = -T_est[:3, :3].T @ T_est[:3, 3]
    c_true = -T_true[:3, :3].T @ T_true[:3, 3]
    cos = ((torch.trace(D[:3, :3]) - 1) / 2).clamp(-1, 1)
    return float((c_est - c_true).norm()), math.degrees(float(torch.arccos(cos)))


def main():
    g = torch.Generator().manual_seed(0)
    scene = make_scene(g)
    f = 0.9 * kWidth
    K = torch.tensor([[f, 0, kWidth / 2], [0, f, kHeight / 2], [0, 0, 1.0]], device=kDevice)
    state = {"deg": 0.0, "T_est": orbit_pose(0.0).clone(), "T_prev": orbit_pose(0.0).clone(), "paused": False,
             "t_err": [], "r_err": [], "render_ms": 0.0, "track_ms": 0.0}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.canvas.manager.set_window_title("mpsplat on Apple GPU (MPS)")
    blank = np.zeros((kHeight, kWidth, 3))
    ims = {
        "rgb": axes[0, 0].imshow(blank),
        "depth": axes[0, 1].imshow(blank[..., 0], cmap="turbo", vmin=2.0, vmax=8.0),
        "alpha": axes[0, 2].imshow(blank[..., 0], cmap="gray", vmin=0, vmax=1),
        "est": axes[1, 0].imshow(blank),
        "err": axes[1, 1].imshow(blank[..., 0], cmap="magma", vmin=0, vmax=0.5),
    }
    titles = {"rgb": "RGB (true camera)", "depth": "depth (m)", "alpha": "alpha",
              "est": "tracked camera's render", "err": "|true - tracked| RGB"}
    for k, ax in zip(["rgb", "depth", "alpha", "est", "err"], [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]):
        ax.set_title(titles[k])
        ax.axis("off")
    fig.colorbar(ims["depth"], ax=axes[0, 1], fraction=0.035)
    ax_plot = axes[1, 2]
    (line_t,) = ax_plot.plot([], [], label="centre error (cm)")
    (line_r,) = ax_plot.plot([], [], label="rotation error (deg)")
    ax_plot.set_xlabel("frame")
    ax_plot.set_title("tracking error")
    ax_plot.legend(loc="upper right")
    ax_plot.grid(alpha=0.3)
    info = fig.suptitle("")

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    def step(_):
        if state["paused"]:
            return
        state["deg"] += kStepDeg
        T_true = orbit_pose(state["deg"])
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            rgb, depth, alpha = render(scene, T_true, K)
        torch.mps.synchronize()
        state["render_ms"] = 1000 * (time.perf_counter() - t0)

        # tracking: predict with constant velocity, then Adam on tau
        t0 = time.perf_counter()
        T = state["T_est"] @ torch.linalg.inv(state["T_prev"]) @ state["T_est"]
        tau = torch.zeros(6, device=kDevice, requires_grad=True)
        opt = torch.optim.Adam([tau], lr=kTrackLr)
        for _ in range(kTrackSteps):
            opt.zero_grad()
            e_rgb, e_depth, _ = render(scene, se3_exp(tau) @ T, K)
            loss = (e_rgb - rgb).abs().mean() + 0.1 * (e_depth - depth).abs().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            state["T_prev"], state["T_est"] = state["T_est"], se3_exp(tau) @ T
            e_rgb, _, _ = render(scene, state["T_est"], K)
        torch.mps.synchronize()
        state["track_ms"] = 1000 * (time.perf_counter() - t0)

        te, re = pose_error(state["T_est"], T_true)
        state["t_err"].append(100 * te)
        state["r_err"].append(re)

        ims["rgb"].set_data(rgb.clamp(0, 1).cpu().numpy())
        ims["depth"].set_data(np.where(alpha.cpu().numpy() > 0.5, depth.cpu().numpy() / alpha.clamp(min=1e-6).cpu().numpy(), np.nan))
        ims["alpha"].set_data(alpha.cpu().numpy())
        ims["est"].set_data(e_rgb.clamp(0, 1).cpu().numpy())
        ims["err"].set_data((e_rgb - rgb).abs().mean(-1).cpu().numpy())
        x = np.arange(len(state["t_err"]))
        line_t.set_data(x, state["t_err"])
        line_r.set_data(x, state["r_err"])
        ax_plot.set_xlim(0, max(10, len(x)))
        ax_plot.set_ylim(0, max(1.0, 1.1 * max(max(state["t_err"]), max(state["r_err"]))))
        info.set_text(f"{scene['means'].shape[0]} Gaussians, {kWidth}x{kHeight}, MPS.   "
                      f"render {state['render_ms']:.1f} ms;  tracking ({kTrackSteps} fwd+bwd steps) "
                      f"{state['track_ms']:.0f} ms;   orbit {state['deg']:.0f} deg, {kStepDeg} deg/frame.   "
                      f"space: pause, q: quit")

    anim = FuncAnimation(fig, step, interval=1, cache_frame_data=False)  # noqa: F841 (keep a reference)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
