import { useEffect, useMemo, useRef, useState } from "react";
import { GRID_SIZES, MU_SET, S_RUN, S_REF } from "./lib/constants";
import { dxFromN, tauFromMu, expectedK } from "./lib/math";

import init, { Solver } from "solver-rust-wasm";

const VIEW_PX = 384;

export default function App() {
  const [N, setN] = useState<number>(64);
  const [alpha, setAlpha] = useState<number>(0.2);
  const [mu, setMu] = useState<number>(10);

  const dx = useMemo(() => dxFromN(N), [N]);
  const tauJs = useMemo(() => tauFromMu(mu, alpha, dx), [mu, alpha, dx]);
  const kRunExp = useMemo(() => expectedK(mu, S_RUN), [mu]);
  const kRefExp = useMemo(() => expectedK(mu, S_REF), [mu]);

  const solverRef = useRef<Solver | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // reuse one offscreen canvas to avoid allocating every draw
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  const [wasmReady, setWasmReady] = useState(false);
  const [tauRust, setTauRust] = useState<number | null>(null);
  const [kUsed, setKUsed] = useState<number | null>(null);
  const [computeMs, setComputeMs] = useState<number | null>(null);
  const [displayMode, setDisplayMode] = useState<"raw" | "auto">("auto");
  const [maxVal, setMaxVal] = useState<number>(0);
  const imageDataRef = useRef<ImageData | null>(null);


  useEffect(() => {
    (async () => {
      await init();
      setWasmReady(true);
    })();
  }, []);

  useEffect(() => {
    if (!wasmReady) return;

    const s = new Solver(N);
    s.set_s_run(S_RUN);
    s.set_alpha(alpha);
    s.set_mu(mu);

    // seed
    s.add_hotspot(Math.floor(N / 2), Math.floor(N / 2), 1.0);

    solverRef.current = s;
    setTauRust(s.get_tau());
    drawHeatmap(s);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wasmReady, N]);

  useEffect(() => {
    const s = solverRef.current;
    if (!s) return;
    s.set_alpha(alpha);
    s.set_mu(mu);
    setTauRust(s.get_tau());
  }, [alpha, mu]);

  function drawHeatmap(s: Solver) {
    const canvas = canvasRef.current;
    if (!canvas) return;
  
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
  
    const n = s.n();
  
    // Robust field read: wasm-bindgen may give Float32Array OR plain Array<number>
    const rawAny: any = s.get_field();
    const field: ArrayLike<number> =
      rawAny instanceof Float32Array ? rawAny :
      Array.isArray(rawAny) ? rawAny :
      rawAny;
  
    // Compute max for diagnostics + auto-contrast
    let mx = 0;
    for (let i = 0; i < field.length; i++) {
      const v = field[i] as number;
      if (v > mx) mx = v;
    }
    setMaxVal(mx);
  
    // Prepare offscreen canvas
    let off = offscreenRef.current;
    if (!off) {
      off = document.createElement("canvas");
      offscreenRef.current = off;
    }
    if (off.width !== n || off.height !== n) {
      off.width = n;
      off.height = n;
      imageDataRef.current = null; // force recreate if resolution changes
    }
  
    const octx = off.getContext("2d");
    if (!octx) return;
  
    // Reuse ImageData buffer (avoids allocations every draw)
    let img = imageDataRef.current;
    if (!img || img.width !== n || img.height !== n) {
      img = octx.createImageData(n, n);
      imageDataRef.current = img;
    }
  
    const data = img.data;
  
    // Choose scaling based on display mode
    const eps = 1e-12;
    const inv = displayMode === "auto" ? 1 / (mx + eps) : 1;
  
    // Gamma makes low values visible (especially in auto mode)
    const gamma = displayMode === "auto" ? 0.35 : 1.0;
  
    for (let i = 0; i < field.length; i++) {
      const raw = (field[i] as number);
  
      // scale
      let v = raw * inv;
  
      // clamp
      if (v < 0) v = 0;
      if (v > 1) v = 1;
  
      // gamma
      if (gamma !== 1.0) v = Math.pow(v, gamma);
  
      const c = (v * 255) | 0;
  
      const j = i * 4;
      data[j + 0] = c;
      data[j + 1] = c;
      data[j + 2] = c;
      data[j + 3] = 255;
    }
  
    octx.putImageData(img, 0, 0);
  
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }
  

  function stepOnce() {
    const s = solverRef.current;
    if (!s) return;

    const info = s.step_tau();
    setKUsed(info.k);
    setComputeMs(info.compute_ms);
    setTauRust(info.tau);

    drawHeatmap(s);
  }

  function resetField() {
    const s = solverRef.current;
    if (!s) return;

    s.clear();
    s.add_hotspot(Math.floor(N / 2), Math.floor(N / 2), 1.0);

    setTauRust(s.get_tau());
    setKUsed(null);
    setComputeMs(null);
    drawHeatmap(s);
  }

  function onCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const s = solverRef.current;
    const canvas = canvasRef.current;
    if (!s || !canvas) return;

    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const n = s.n();
    const x = Math.floor((px / rect.width) * n);
    const y = Math.floor((py / rect.height) * n);

    s.add_hotspot(x, y, 1.0);
    drawHeatmap(s);
  }

  return (
    <div style={{ padding: 16, fontFamily: "system-ui" }}>
      <h1>Thermal-AI (v1 scaffold)</h1>

      <div style={{ display: "grid", gap: 12, maxWidth: 720 }}>
        <div style={{ display: "grid", gap: 8, gridTemplateColumns: "auto 1fr" }}>
          <label>Grid N:</label>
          <select value={N} onChange={(e) => setN(Number(e.target.value))}>
            {GRID_SIZES.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>

          <label>α (diffusivity):</label>
          <input
            type="number"
            value={alpha}
            min={0.05}
            max={0.5}
            step={0.01}
            onChange={(e) => setAlpha(Number(e.target.value))}
            style={{ width: 140 }}
          />

          <label>μ (jump):</label>
          <select value={mu} onChange={(e) => setMu(Number(e.target.value))}>
            {MU_SET.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <button onClick={stepOnce} disabled={!wasmReady}>
            Step τ
          </button>
          <button onClick={resetField} disabled={!wasmReady}>
            Reset
          </button>
          <div>WASM: {wasmReady ? "ready" : "loading..."}</div>
        </div>

        <canvas
          ref={canvasRef}
          width={VIEW_PX}
          height={VIEW_PX}
          onClick={onCanvasClick}
          style={{ border: "1px solid #444", cursor: "crosshair" }}
        />

        <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
          Display:
          <select value={displayMode} onChange={(e) => setDisplayMode(e.target.value as any)}>
            <option value="auto">Auto-contrast</option>
            <option value="raw">Raw (0..1)</option>
          </select>
        </label>


        <div style={{ padding: 12, border: "1px solid #444", borderRadius: 8 }}>
          <div>dx (JS) = {dx.toExponential(4)}</div>
          <div>τ (JS) = μ·dx²/α = {tauJs.toExponential(4)} s</div>
          <div>τ (Rust) = {tauRust !== null ? tauRust.toExponential(4) : "-"} s</div>
          <div>Expected k_run (s={S_RUN}) ≈ {kRunExp}</div>
          <div>Expected k_ref (s={S_REF}) ≈ {kRefExp}</div>
          <div style={{ marginTop: 8 }}>
            k used = {kUsed ?? "-"} | solver compute ={" "}
            {computeMs !== null ? `${computeMs.toFixed(3)} ms` : "-"}
          </div>
          <div>max(u) = {maxVal.toExponential(3)}</div>
          <div style={{ marginTop: 6, fontSize: 13, opacity: 0.85 }}>
            Tip: click on the canvas to add hotspots, then Step τ.
          </div>
        </div>
      </div>
    </div>
  );
}
