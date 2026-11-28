# VIctory-Fast

A performance-focused fork of [VIctory](https://github.com/Marvisak/VIctory), a Direct3D 12 and runtime compatibility layer for running Xbox Series X|S binaries natively on Windows using files supplied by the user.

**VIctory-Fast** targets **40+ FPS playable raster performance on AMD RDNA 4 / Radeon RX 9070 XT** (and modern PC GPUs) through native PC subsystems, RDNA 4 Wave32 shader lowering, lock-free submission/shadow paths, GPU compute detiling & BCPack, native PC FSR 4.1.2 / HDR10 / VRR / GameInput, an **in-game F3/F4 performance HUD & live debug menu**, **and** optional **native hardware DirectX Raytracing (DXR 1.1 `RayQuery`)**.

> **Legal & Asset Policy**
>
> **This repository contains no copyrighted game files, executables, shaders, textures, audio, or assets.** You must supply your own legally obtained dump of any title you run. This project is an independent open-source compatibility layer under **GPL-3.0** and is not affiliated with or endorsed by any console manufacturer, game developer, or publisher. See [docs/asset-policy.md](docs/asset-policy.md).

---

## Key Features

### 40+ FPS RDNA 4 & Native PC Performance
- **RDNA 4 Wave32 Shader Lowering & Codegen (`VICTORY_RDNA_WAVE32=2`, `VICTORY_SHADER_OPT=127`, `VICTORY_LEGACY_MUL_PLAIN=1`):** Tuned DXIL emission for Radeon RX 9070 / 9070 XT occupancy (narrowed barriers, fast reciprocal, RDNA peepholes, optional plain legacy mul/fmad) without VGPR-bloat regressions, paired with `VICTORY_LDS_INIT_HOIST` for clean water/compute initialization.
- **Lock-Free Submission, Barriers & Memory Architecture:** Relaxed Format Casting (`CreateCommittedResource3`), Enhanced Barriers + deferred-fence Async Compute DXR prelude, frame-ring command allocators (`498 → ~48` live), linear upload ring, lock-free `kernel_writing` + 64-shard `GUARDS`, hardware-tripwired Mode-1 GPU-only copyback (`VICTORY_GPU_ONLY_COPYBACK=1`, skipping `16.4 GB/f` GPU writes with `0` trips), chunked `SEC_RESERVE` + direct I/O (`OS page faults 254,000 → 759/f`), zero-retention shader cache + isolated `--compile-worker` processes, demand-tiered BAR, and uncontended `list::thin` CBV binding (**`42.1–42.9 FPS`** RT-Off / **`34.6–35.2 FPS`** Native DXR 1.1 at Series X `1080p → 1440p` PC FSR 4.1.2 with `RAM 21.1–22.6 GB`, `VRAM 10.1–10.9 GB`, `SHR 2.65–2.79 GB`, and `0` faults).
- **In-Game Performance HUD & Live Debug Menu (`F3` / `F4`):** Zero-overhead when hidden (`--release` and `diag`). Press **`F3`** to cycle `Off → Compact → Full` HUD (FPS, 1% Low, rolling 120-frame frametime graph with 60/30 FPS guides, RAM/VRAM bars, SDR/HDR10/RGBA16F aware), and **`F4`** for an arrow-key live debug menu (HUD mode, frametime graph toggle & scale, corner, UI scale, live `VICTORY_AMBIENT` slider `0.0–3.0`, and stats reset).
- **Native PC Subsystems:**
  - **GPU Compute Detiler & BCPack Decoder:** Offloads tiled texture detiling and BCPack decompression onto D3D12 compute shaders with dirty-page partial uploads.
  - **Native PC Display & FidelityFX (`FSR 4.1.2` + opt-in Frame Gen):** DXGI tearing/VRR, waitable swapchain, HDR10 output, automatic FidelityFX upscaler selection (`1080p → 1440p` FSR 4.1.2), and opt-in Frame Generation bridge (`VICTORY_FFX_FG=1`).
  - **In-Place Memory & Render Targets (`VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORT_RT=1`):** Zero-copy GPU reads of title constant buffers and in-place render targets/heaps with zero-syscall memory probes.
  - **Native `GameInput.dll` & Audio:** Mouse/keyboard, impulse triggers, XMA overlap persistence, and native audio resampling.
  - **RT-Off Visual Fixes:** Clamps indoor RT-off sky-miss reflections (`CS_UpdateHistory` indirect hook + `VICTORY_AMBIENT_RT`) so eyes, framed pictures, and indoor specular surfaces render naturally without skybox bleed.

### Native Hardware DXR 1.1 Ray Tracing (`VICTORY_NATIVE_RT=1`)
- **`XboxBvhBuilder.dll` HLE (`crates/victory-hle/src/modules/bvh_builder.rs`):** Intercepts console BVH builder APIs directly rather than emulating dozens of BVH-building compute shaders in software.
- **Host DXR Acceleration Structure Manager (`crates/victory-gpu/src/dxr.rs`):** Builds and updates native D3D12 multi-geometry static BLASes, dynamic BLASes (with BLAS refitting & scratch batching), and `PREFER_FAST_TRACE` TLASes directly on host RT hardware with topology validation, safe deferred resource retention, and null-TLAS fallback handling.
- **RDNA 2 → DXIL SM 6.6 `RayQuery` Lifter (`crates/victory-shader/src/emit/rayquery_lift.rs`, `lower.rs`):** Detects `image_bvh*_intersect_ray` traversal loops (opaque, front/back-track, and alpha-tested any-hit loops), coalesces `AllocateRayQuery` handles across disjoint loops, and lifts them into native DXIL Shader Model 6.6 `RayQuery` (`TraceRayInline` / `Proceed` / `Committed*` / `Candidate*`) instructions. Closest-hit `AnyHit` loops default to the inline fallback path on RDNA 4 (opt-in via `VICTORY_NATIVE_RT_ANYHIT=1`) so opaque/shadow passes capture the full hardware speedup without divergent `Proceed()` stalls.
- **AMD RDNA 4 Stability (`amdxc64.dll`):** Fixes buffer `getDimensions` lowering crashes and validates lifted DXR shaders through `dxil.dll` and host D3D12 PSO creation.

---

## Performance & Benchmarks

Measured on **AMD Ryzen 7 7800X3D · AMD Radeon RX 9070 XT (16 GB, RDNA 4) · 32 GB RAM · Windows 11**.

### Configuration FPS Summary (Measured)

| Configuration | Build | Median / Typical FPS | Peak FPS | Notes |
|---|---|---:|---:|---|
| **Series X (1440p, FSR 4.1.2) — RT Off** | `--release` (measured, `281aacf`) | **~42.1–42.9 fps** settled (`p1 17.2 fps`, `p99 47.1 ms`) | **43.4+ fps** | `1080p → 1440p` PC FSR 4.1.2 + `VICTORY_AMBIENT=1.4` + Mode-1 tripwired copyback (`5.1–5.5 ms`, `4.3 MB/f`, `7/7` armed, `0` tripped): **`23.3 ms/f`**, `list::thin 82–88 ms` (`-63%`), **`21.05 GB`** RAM (`13.67 GB` WS), **`10.10 GB`** VRAM (`65.5%`), **`2.79 GB`** SHR, `759` faults/f, `0` GPU faults |
| **Series S (1080p) — RT Off** | `--release` (measured, `7d2f44f`) | **~30.4 fps** settled (`28.6` overall med) | **32.4 fps** | `VICTORY_RDNA_WAVE32=2`: `32.9 ms/f`, **`23.9 ms`** prep, **`0.0 ms`** GPU wait, `14.7 ms` copyback, `16.7 GB` RAM / `8.2 GB` VRAM / `2.09 GB` SHR, `0` faults |
| **Series X (Native 1440p) — RT Off** | `--release` (measured, `7d2f44f`) | **~29.8 fps** settled (`30.7` final) | **30.8 fps** | `VICTORY_RDNA_WAVE32=2`: `33.6 ms/f`, **`20.8 ms`** prep, **`0.0 ms`** GPU wait, `13.9 ms` copyback, `19.5 GB` RAM / `9.6 GB` VRAM / `1.75 GB` SHR, `0` faults |
| **Series S — Native DXR 1.1 RT** | `--release` (measured, `78b3b7b`, parity default) | **~34.6–34.8 fps** settled (`p1 23.0 fps`, `max 43.5 ms`) | **34.8+ fps** | `28.8 ms/f`, lifted `SLOT_DXR_NORMALS` + two-pass `tMax` pre-shrink + mask-split `FrontBackTrack` + chunked `let_go_idle`, `0` faults |
| **Series S — Software RT** | `--release` (measured, `d1686f1`, 165s) | **~11.7 fps** settled (`12.0 fps` overall med) | **18.8 fps** | `1.4 ms` med / `4.7 ms` settled GPU wait, `54.4 ms` prep, `11.3 GB` max VRAM (`128.6 ms` timed `diag`) |
| **Series X (1440p) — Software RT** | `--release` (measured) | **~8.2 fps** | **19.9 fps** | Software BVH emulation (`175.8 ms` GPU in timed `diag`) |
| **Series X (1440p, FSR 4.1.2) — Native DXR 1.1 RT** | `--release` (measured, `281aacf`, `VICTORY_AMBIENT=0`) | **~34.6–35.2 fps** settled (`p1 17.8 fps`, `max 56.1 ms`) | **35.2+ fps** | `1080p → 1440p` PC FSR 4.1.2 + un-fused speculative FNO + 2-pass Mirror TLAS + direct-normal coalescing + Mode-1 tripwired copyback (`7.2 ms`, `9.0 MB/f`, `12/12` armed, `0` tripped, skipping `16.4 GB/f`) + Async Compute DXR prelude + D3D12 Debug Layer `0` errors: **`28.4–28.5 ms/f`**, **`22.58 GB`** RAM (`14.47 GB` WS), **`10.91 GB`** VRAM (`70.7%`), **`2.65 GB`** SHR, `0` dropped draws/PSOs, `0` GPU faults |

### Native DXR 1.1 vs. Software BVH (Head-to-Head `VICTORY_GPU_TIMING=1` + `DRED`, Series S Profile)

| Metric | Software BVH (`VICTORY_BVH_BUILDER=1`) | Native DXR 1.1 (`VICTORY_NATIVE_RT=1`) | Delta |
|---|---:|---:|---:|
| Total GPU card time / frame | `128.6 ms` | **`80.6 ms`** | **-37.3% (-48.0 ms)** |
| Indirect compute time / frame | `105.4 ms` | **`64.2 ms`** | **-39.1% (-41.2 ms)** |
| Shadow RT pass (`f607780aa8e502d0`) | `8.20 ms` | **`~0.1 ms`** | **~80× faster** |
| RT pass (`caf96a3229145fa6`) | `21.04 ms` | **`10.39 ms`** | **-50.6%** |
| RT pass (`36f04a4ed0ad1d9b`) | `5.01 ms` | **`2.19 ms`** | **-56.3%** |
| RT pass (`3592426e3e9142d8`) | `6.01 ms` | **`3.30 ms`** | **-45.1%** |
| Resident VRAM / faults | ~12–16 GB | **`~10.2 GB` (`0` faults)** | **~2+ GB lower** |

*Note: `diag` runs with `VICTORY_GPU_TIMING=1` and `VICTORY_D3D12_DRED=1` incur heavy instrumentation overhead (~3–6 fps) and are used strictly for apples-to-apples per-shader GPU timing. Use `--release` without timing/DRED flags for gameplay.*

### Renderer, Residency & Frame-Time Improvements (Inherited from Upstream Forks)
- **Async list recording & parallel submission prep:** `VICTORY_THIN`, `VICTORY_ASYNC_SUBMIT`, `VICTORY_JOIN_PARTS`, AVX2 page comparisons, and parallel descriptor/refresh passes.
- **Background GPU read-back & on-card steps:** `VICTORY_RELAY`, `VICTORY_LAND`, `VICTORY_CARD_STEPS`, and texture read-back.
- **32 GB RAM residency & two-plane composition:** Fallback heaps, commit-pressure backoff, `TablePlan` caching, Wave32 lowering support, and Series S two-plane composition.
- **Frame record & replay diagnostics:** Per-pass GPU timing (`VICTORY_GPU_TIMING=1`) and replay tooling ([docs/replay.md](docs/replay.md)).

---

## Requirements

- **OS:** Windows 10/11 x64
- **CPU:** x86-64 with AVX2
- **GPU:** Direct3D 12 GPU with **Shader Model 6.6**, **Mesh Shaders**, and **DXR Tier 1.1** (for `VICTORY_NATIVE_RT=1`)
- **Build tools:** Rust toolchain (pinned in `rust-toolchain.toml`) and MSVC C++ Build Tools with Windows SDK
- **Title files:** Your own dump stored **outside** this repository

---

## Building & Running

```powershell
# Release build (recommended for gameplay)
cargo build --release --locked -p victory

# Diagnostics build (with frame/GPU timing instrumentation)
cargo build --profile diag --locked -p victory --features diagnostics
```

Configure local paths outside the repository (see `victory.example.toml`):

```powershell
copy victory.example.toml victory.toml
```

Run with the PowerShell launcher:

```powershell
# Standard run
scripts\play.ps1 -Exe <path-to-your-title-executable> -Build release

# With ray tracing + native DXR 1.1 hardware acceleration
scripts\play.ps1 -Exe <path-to-your-title-executable> -Build release -RayTracing -NativeRT

# Safe fallback mode (disables experimental optimizations)
scripts\play.ps1 -Exe <path-to-your-title-executable> -Safe
```

Or with `Play-VIctory.cmd`:

```powershell
$env:VICTORY_GAME_EXE = 'D:\Games\MyDump\title.exe'
.\Play-VIctory.cmd native-rt
```

Optional ahead-of-time shader/pipeline warm-up:

```powershell
target\release\victory.exe shaders <path-to-shader-archives>
target\release\victory.exe warm --used
```

---

## Key Environment Switches

| Variable | Effect |
|---|---|
| `VICTORY_RDNA_WAVE32=2` | RDNA 4 Wave32 shader lowering for higher RX 9070 / 9070 XT occupancy |
| `VICTORY_LEGACY_MUL_PLAIN=1` | Plain IEEE mul/fmad on AMD RDNA (`+2.6 FPS`, enabled in non-Safe `scripts/play.ps1`) |
| `VICTORY_PANEL=1` / `2` | Start with in-game Performance HUD (`1` Compact, `2` Full; toggle anytime with `F3`, debug menu with `F4`) |
| `VICTORY_FFX_FG=1` | Opt-in FidelityFX Frame Generation bridge |
| `VICTORY_PORT_RT=1` | In-place render target placement to reduce copyback & memory traffic |
| `VICTORY_LDS_INIT_HOIST=1` | Hoist LDS initialization for clean water & compute shaders |
| `VICTORY_AMBIENT_RT=1` | RT-off ambient/sky-miss reflection clamping for natural indoor lighting |
| `VICTORY_NATIVE_RT=1` | Enable native D3D12 DXR 1.1 `RayQuery` lifting & host AS management |
| `VICTORY_NATIVE_RT_ANYHIT=1` | Force closest-hit `AnyHit` loops onto `RayQuery` `Proceed()` (default off on RDNA 4) |
| `VICTORY_BVH_BUILDER=1` | Enable BVH builder support (paired with `VICTORY_NATIVE_RT=1` or software fallback) |
| `VICTORY_D3D12_DRED=1` / `light` | Enable D3D12 DRED breadcrumbs for GPU fault diagnostics |
| `VICTORY_GPU_TIMING=1` | Log per-command GPU execution times (pair with `diag` build + DRED) |
| `VICTORY_CACHE_DIR` | Override local shader/pipeline cache directory (default under `%LOCALAPPDATA%`) |

For additional switches and architecture notes, see [docs/architecture.md](docs/architecture.md), [docs/fork-ray-tracing.md](docs/fork-ray-tracing.md), [docs/findings.md](docs/findings.md), and [docs/README-fxceless.md](docs/README-fxceless.md).

---

## Credits & License

Licensed under the **GNU General Public License v3.0** ([LICENSE](LICENSE)).

Built upon the work of the VIctory community:
- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)** — core loader, runtime HLE, D3D12 backend, and shader recompiler.
- **fxceless**, **avatarmanz**, **codec-d**, **DexNom**, **Belobug**, **romainhedouin**, **ZJOU**, **myokex** (with Spexivus, joroyt, tightfall), **Joshua Odom**, **Geckomacabre**, and **cxy0072019-sketch** — upstream renderer, residency, read-back, Wave32, and AMD/NVIDIA improvements.
