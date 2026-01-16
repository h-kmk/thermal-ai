# Thermal-AI

Hybrid neural–numerical **2D heat diffusion** simulator in the browser.

The goal is to demonstrate **measurable wall-clock speedups** by learning a **μ-conditioned τ-jump diffusion operator** (temporal super-resolution), while tracking the speed–accuracy–physics tradeoffs under a fixed experimental protocol.

## What you can do right now
- Run a Rust→WASM solver in the browser
- Click to add hotspots, then advance the field by a **τ jump**
- See the solver’s **k substeps** and **compute time** per jump

## Repository structure
- `solver-core/` — Pure Rust diffusion solver logic (no WASM, no UI)
- `solver-rust-wasm/` — WASM bindings + browser-facing wrapper (wasm-bindgen / wasm-pack)
- `solver-cli/` — CLI for dataset generation (writes `input.bin`, `target.bin`, `meta.jsonl`)
- `web-ui/` — Vite + React UI demo
- `ml/` — Training/eval (planned): U-Net baseline, physics-informed variant, ONNX export
- `benchmarks/` — Benchmark logs/plots (planned)

## Protocol
See [`protocol.md`](protocol.md). This defines:
- fixed constants (`N`, `μ` set, `α` range, `s_run`, `s_ref`)
- how τ is derived from μ
- baselines, metrics, OOD tests, reproducibility requirements

## Milestone status (current)
**M1**: Repo scaffold + Vite UI + Rust/WASM wiring  
**M2**: τ-jump stable stepping with `k = ceil(τ / (s * dt_max))`  
**M3**: Split solver into `solver-core` + WASM wrapper + CLI dataset generator  

**Next (M4)**: Expand dataset generator (multiple IC families, per-trajectory seeds, train/val/test splits)  
**Then (M5)**: Train baseline U-Net on (uᵗ, μ, x, y) → uᵗ⁺ᵗᵃᵘ, export ONNX  
**Then (M6)**: Browser inference + end-to-end speed/accuracy benchmarking

## Quick dev notes
- Build WASM: `wasm-pack build --target web --out-dir pkg` (inside `solver-rust-wasm/`)
- UI runs in `web-ui/` (Vite)
- Dataset generation lives in `solver-cli/` (see `solver-cli --help`)
