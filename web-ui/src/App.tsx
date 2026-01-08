import { useMemo, useState } from "react";
import { GRID_SIZES, MU_SET, S_RUN, S_REF } from "./lib/constants";
import { dxFromN, tauFromMu, expectedK } from "./lib/math";

export default function App() {
  const [N, setN] = useState<number>(64);
  const [alpha, setAlpha] = useState<number>(0.2);
  const [mu, setMu] = useState<number>(10);

  const dx = useMemo(() => dxFromN(N), [N]);
  const tau = useMemo(() => tauFromMu(mu, alpha, dx), [mu, alpha, dx]);
  const kRun = useMemo(() => expectedK(mu, S_RUN), [mu]);
  const kRef = useMemo(() => expectedK(mu, S_REF), [mu]);

  return (
    <div style={{ padding: 16, fontFamily: "system-ui" }}>
      <h1>Thermal-AI (v1 scaffold)</h1>

      <div style={{ display: "grid", gap: 12, maxWidth: 520 }}>
        <label>
          Grid N:
          <select value={N} onChange={(e) => setN(Number(e.target.value))}>
            {GRID_SIZES.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </label>

        <label>
          α (diffusivity):
          <input
            type="number"
            value={alpha}
            min={0.05}
            max={0.5}
            step={0.01}
            onChange={(e) => setAlpha(Number(e.target.value))}
            style={{ marginLeft: 8, width: 100 }}
          />
        </label>

        <label>
          μ (jump):
          <select value={mu} onChange={(e) => setMu(Number(e.target.value))}>
            {MU_SET.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </label>

        <div style={{ padding: 12, border: "1px solid #444", borderRadius: 8 }}>
          <div>dx = {dx.toExponential(4)}</div>
          <div>τ = μ·dx²/α = {tau.toExponential(4)} s</div>
          <div>Expected k_run (s={S_RUN}) ≈ {kRun}</div>
          <div>Expected k_ref (s={S_REF}) ≈ {kRef}</div>
        </div>
      </div>
    </div>
  );
}
