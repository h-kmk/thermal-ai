use wasm_bindgen::prelude::*;
use solver_core::SolverCore;

#[wasm_bindgen]
pub struct Solver {
    inner: SolverCore,
}

#[wasm_bindgen]
impl Solver {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Result<Solver, JsValue> {
        let inner = SolverCore::new(n).map_err(|e| JsValue::from_str(&e))?;
        Ok(Solver { inner })
    }

    // Parameters
    pub fn set_alpha(&mut self, alpha: f32) { self.inner.set_alpha(alpha); }
    pub fn set_mu(&mut self, mu: f32) { self.inner.set_mu(mu); }
    pub fn set_s_run(&mut self, s: f32) { self.inner.set_s_run(s); }
    pub fn set_s_ref(&mut self, s: f32) { self.inner.set_s_ref(s); }

    pub fn get_dx(&self) -> f32 { self.inner.get_dx() }
    pub fn get_tau(&self) -> f32 { self.inner.get_tau() }

    pub fn clear(&mut self) { self.inner.clear(); }
    pub fn add_hotspot(&mut self, x: usize, y: usize, value: f32) {
        self.inner.add_hotspot(x, y, value);
    }

    pub fn n(&self) -> usize { self.inner.n() }

    // Copy-based JS access (reliable)
    pub fn get_field(&self) -> Vec<f32> {
        self.inner.field().to_vec()
    }

    // Step + timing (WASM-only)
    pub fn step_tau(&mut self) -> StepInfo {
        let t0 = now_ms();
        let (k, tau) = self.inner.step_tau_run();
        let t1 = now_ms();
        StepInfo { k, compute_ms: t1 - t0, tau }
    }
    pub fn step_tau_ref(&mut self) -> StepInfo {
        let t0 = now_ms();
        let (k, tau) = self.inner.step_tau_ref();
        let t1 = now_ms();
        StepInfo { k, compute_ms: t1 - t0, tau }
    }
    
}

#[wasm_bindgen]
pub struct StepInfo {
    k: u32,
    compute_ms: f64,
    tau: f32,
}

#[wasm_bindgen]
impl StepInfo {
    pub fn k(&self) -> u32 { self.k }
    pub fn compute_ms(&self) -> f64 { self.compute_ms }
    pub fn tau(&self) -> f32 { self.tau }
}


fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}
