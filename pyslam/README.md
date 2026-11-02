# pySLAM's checks of mpsplat

pySLAM (github.com/sjulier/pyslam) will render its Gaussian splatting maps (MonoGS) through gsplat's API:
conda-forge's `gsplat` on CUDA, and this fork of mpsplat on Apple GPUs. `reference.py` checks that the
two agree. It uses the calls pySLAM will make: `rasterization(..., render_mode="RGB+D", packed=False)`,
and keyframe poses refined by autograd through the view matrix, `viewmat = SE3_exp(tau) @ T`, with
MonoGS's `SE3_exp`.

`reference_cuda.npz` holds two fixed scenes (50 Gaussians at 64x48, and 5000 at 320x240). For each scene
it stores the RGB, the accumulated depth and the alpha, and the gradients of a fixed linear loss with
respect to every input: means, quaternions, log-scales, opacity logits, colours, the pose update tau,
and the 2D means. It was made with gsplat 1.5.3 (conda-forge), torch 2.12.1 and CUDA 12.9 on an
RTX 5090.

```
python pyslam/reference.py compare --device mps   # against the CUDA reference
python pyslam/reference.py fd --device mps        # finite differences, needs no reference
python pyslam/reference.py speed --device mps     # forward+backward ms at 640x480, 100k and 300k Gaussians
```

On CUDA (RTX 5090), 2 Nov 2026:
- `fd`: every gradient agrees with central differences to 3-4 digits; the pose gradient agrees to
  within 3-15% (float32 with a 1e-3 step).
- `speed`: 4.2 ms with 100k Gaussians and 9.5 ms with 300k.

Use accumulated depth (`RGB+D`), not expected depth (`RGB+ED`), in losses. Expected depth is divided by
the alpha, so it jumps at the edge of every Gaussian.
