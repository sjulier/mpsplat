// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 3DGS backward tile rasterizer.
//
// Walks the same gaussians as the forward kernel, in reverse depth order,
// recomputes alpha to match forward, derives per-pixel gradient
// contributions, and atomic-adds them into per-gaussian gradient buffers.
//
// Differences from CUDA:
//   * The gradients of each gaussian are summed over the SIMD group
//     (simd_sum) before one atomic add, like CUDA's warp sum.
//   * No threadgroup cache yet — each thread reads gaussian state directly
//     from device memory. Stage 7.x can add a cache once the baseline is
//     verified to match the torch reference within tolerance.
//   * `MAX_ALPHA` and `ALPHA_THRESHOLD` semantics match the *forward*
//     kernel's relaxed flavour (no alpha-threshold cull, no early-exit) —
//     the forward stays paired with its bwd; if Stage 7.x reintroduces
//     CUDA's optimisations both kernels move together.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint  TILE_SIZE              = 16u;
constant constexpr uint  BLOCK_SIZE             = TILE_SIZE * TILE_SIZE;
constant constexpr float MAX_ALPHA              = 0.99f;
constant constexpr float MIN_ONE_MINUS_ALPHA    = 1e-7f;

kernel void rasterize_to_pixels_3dgs_bwd(
    // gradient outputs (atomic — one per leaf input)
    device atomic_float*        v_means2d        [[buffer(0)]], // [I*N, 2] flat
    device atomic_float*        v_conics         [[buffer(1)]], // [I*N, 3] flat
    device atomic_float*        v_colors         [[buffer(2)]], // [I*N, CDIM] flat
    device atomic_float*        v_opacities      [[buffer(3)]], // [I*N] flat
    // forward inputs
    device const packed_float2* means2d          [[buffer(4)]],
    device const packed_float3* conics           [[buffer(5)]],
    device const float*         colors           [[buffer(6)]], // [I*N, CDIM]
    device const float*         opacities        [[buffer(7)]], // [I*N]
    // forward outputs needed for the backward
    device const float*         render_alphas    [[buffer(8)]], // [I, H, W]
    device const int*           last_ids         [[buffer(9)]], // [I, H, W]
    // upstream gradients
    device const float*         v_render_colors  [[buffer(10)]], // [I, H, W, CDIM]
    device const float*         v_render_alphas  [[buffer(11)]], // [I, H, W]
    // intersections
    device const int*           tile_offsets     [[buffer(12)]], // [I, th, tw]
    device const int*           flatten_ids      [[buffer(13)]], // [n_isects]
    // shape
    constant uint&              image_width      [[buffer(14)]],
    constant uint&              image_height     [[buffer(15)]],
    constant uint&              tile_width       [[buffer(16)]],
    constant uint&              tile_height      [[buffer(17)]],
    constant uint&              n_isects         [[buffer(18)]],
    constant uint&              num_images       [[buffer(19)]],
    uint3                       gid              [[thread_position_in_grid]],
    uint3                       tg_id            [[threadgroup_position_in_grid]]
) {
    // CDIM is set when the shader is compiled (see `_load_cdim` in _kernels.py).
#ifndef CDIM_VALUE
#define CDIM_VALUE 3
#endif
    constexpr uint CDIM = CDIM_VALUE;

    const uint image_id = gid.z;
    const uint tile_id = tg_id.y * tile_width + tg_id.x;
    const uint pix_x = gid.x;
    const uint pix_y = gid.y;
    // No early return for pixels outside the image: the SIMD-group sums
    // below need every lane. Those pixels have bin_final = -1, so they never
    // contribute.
    const bool inside = (pix_x < image_width) && (pix_y < image_height);
    const uint pix_id = inside ? pix_y * image_width + pix_x : 0u;
    const uint pix_base = image_id * image_height * image_width + pix_id;

    const uint tile_offset_idx = image_id * tile_height * tile_width + tile_id;
    int range_start = tile_offsets[tile_offset_idx];
    int range_end;
    bool is_last_tile = (image_id == num_images - 1)
                        && (tile_id == tile_width * tile_height - 1);
    if (is_last_tile) {
        range_end = (int)n_isects;
    } else {
        range_end = tile_offsets[tile_offset_idx + 1];
    }

    // Per-pixel state from the forward.
    const float T_final = inside ? 1.0f - render_alphas[pix_base] : 1.0f;
    float       T = T_final;
    float       buffer[CDIM];
    for (uint k = 0; k < CDIM; ++k) buffer[k] = 0.0f;
    const int   bin_final = inside ? last_ids[pix_base] : -1;

    // Upstream gradient for this pixel.
    float v_render_c[CDIM];
    for (uint k = 0; k < CDIM; ++k) {
        v_render_c[k] = inside ? v_render_colors[pix_base * CDIM + k] : 0.0f;
    }
    const float v_render_a = inside ? v_render_alphas[pix_base] : 0.0f;

    const float px = (float)pix_x + 0.5f;
    const float py = (float)pix_y + 0.5f;

    // Iterate back-to-front through the gaussians this pixel composited
    // (idx <= bin_final), so that T_n = T_{n+1} / (1 - alpha_n) walks from
    // the outermost to the innermost. All lanes of a SIMD group walk the same
    // gaussians, starting from the last one any lane composited, so the
    // gradients of one gaussian can be summed over the group and written with
    // one atomic add, as CUDA's warp sum does.
    const int idx_first = simd_max(bin_final);
    for (int idx = idx_first; idx >= range_start; --idx) {
        const bool valid = idx <= bin_final;
        if (!simd_any(valid)) {
            continue;
        }
        const int g = flatten_ids[idx];
        const float2 xy = means2d[g];
        const float3 cn = conics[g];
        const float opac = opacities[g];
        const uint c_off = (uint)g * CDIM;

        float v_rgb_local[CDIM];
        for (uint k = 0; k < CDIM; ++k) v_rgb_local[k] = 0.0f;
        float v_opacity_local = 0.0f;
        float v_xy_local_x = 0.0f, v_xy_local_y = 0.0f;
        float v_conic_local_x = 0.0f, v_conic_local_y = 0.0f, v_conic_local_z = 0.0f;

        if (valid) {
            const float dx = xy.x - px;
            const float dy = xy.y - py;
            const float sigma = 0.5f * (cn.x * dx * dx + cn.z * dy * dy)
                              + cn.y * dx * dy;
            const float vis = exp(-sigma);
            float alpha = min(MAX_ALPHA, opac * vis);
            if (sigma < 0.0f) {
                alpha = 0.0f;
            }

            // Rewind transmittance: T_at_g = T_after_g / (1 - alpha)
            const float ra = 1.0f / max(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
            T *= ra;
            const float fac = alpha * T;

            float v_alpha = 0.0f;
            for (uint k = 0; k < CDIM; ++k) {
                v_rgb_local[k] = fac * v_render_c[k];
                v_alpha += (colors[c_off + k] * T - buffer[k] * ra) * v_render_c[k];
            }
            v_alpha += T_final * ra * v_render_a;

            if (opac * vis <= MAX_ALPHA) {
                const float v_sigma = -opac * vis * v_alpha;
                v_conic_local_x = 0.5f * v_sigma * dx * dx;
                v_conic_local_y =        v_sigma * dx * dy;
                v_conic_local_z = 0.5f * v_sigma * dy * dy;
                v_xy_local_x = v_sigma * (cn.x * dx + cn.y * dy);
                v_xy_local_y = v_sigma * (cn.y * dx + cn.z * dy);
                v_opacity_local = vis * v_alpha;
            }

            // Update the running back-buffer BEFORE the next iteration.
            for (uint k = 0; k < CDIM; ++k) buffer[k] += colors[c_off + k] * fac;
        }

        // Sum over the SIMD group; its first lane writes.
        for (uint k = 0; k < CDIM; ++k) {
            const float sum = simd_sum(v_rgb_local[k]);
            if (simd_is_first()) {
                atomic_fetch_add_explicit(v_colors + c_off + k, sum, memory_order_relaxed);
            }
        }
        const float s_c0 = simd_sum(v_conic_local_x);
        const float s_c1 = simd_sum(v_conic_local_y);
        const float s_c2 = simd_sum(v_conic_local_z);
        const float s_x  = simd_sum(v_xy_local_x);
        const float s_y  = simd_sum(v_xy_local_y);
        const float s_o  = simd_sum(v_opacity_local);
        if (simd_is_first()) {
            atomic_fetch_add_explicit(v_conics + (uint)g * 3 + 0, s_c0, memory_order_relaxed);
            atomic_fetch_add_explicit(v_conics + (uint)g * 3 + 1, s_c1, memory_order_relaxed);
            atomic_fetch_add_explicit(v_conics + (uint)g * 3 + 2, s_c2, memory_order_relaxed);
            atomic_fetch_add_explicit(v_means2d + (uint)g * 2 + 0, s_x, memory_order_relaxed);
            atomic_fetch_add_explicit(v_means2d + (uint)g * 2 + 1, s_y, memory_order_relaxed);
            atomic_fetch_add_explicit(v_opacities + (uint)g, s_o, memory_order_relaxed);
        }
    }
}
