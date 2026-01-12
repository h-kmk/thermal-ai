use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Solver {
    n: usize,
    alpha: f32,
    mu: f32,
    s_run: f32,
    dx: f32,
    field: Vec<f32>,
    next: Vec<f32>,
}

#[wasm_bindgen]
impl Solver {
        // ---- Simple, reliable JS access (copy) ----
        pub fn get_field(&self) -> Vec<f32> {
            self.field.clone()
        }
    
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize) -> Result<Solver, JsValue> {
        if n < 3 {
            return Err(JsValue::from_str("n must be >= 3"));
        }
        let dx = 1.0f32 / ((n - 1) as f32);
        let size = n * n;

        Ok(Solver {
            n,
            alpha: 0.2,
            mu: 10.0,
            s_run: 0.8,
            dx,
            field: vec![0.0; size],
            next: vec![0.0; size],
        })
    }

    // ---- Parameters (protocol-aligned) ----

    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha.max(1e-8);
    }

    pub fn set_mu(&mut self, mu: f32) {
        self.mu = mu.max(0.0);
    }

    pub fn set_s_run(&mut self, s: f32) {
        // keep it sane; you can tighten later
        self.s_run = s.clamp(0.05, 0.99);
    }

    pub fn get_dx(&self) -> f32 {
        self.dx
    }

    pub fn get_tau(&self) -> f32 {
        // tau = mu * dx^2 / alpha
        (self.mu * self.dx * self.dx) / self.alpha
    }

    pub fn clear(&mut self) {
        self.field.fill(0.0);
        self.next.fill(0.0);
    }

    // Paint a hotspot (simple for now). value is added and then clipped later.
    pub fn add_hotspot(&mut self, x: usize, y: usize, value: f32) {
        if x >= self.n || y >= self.n {
            return;
        }
        let idx = y * self.n + x;
        self.field[idx] = (self.field[idx] + value).clamp(0.0, 1.0);
        self.apply_dirichlet_bc();
    }

    // ---- Field access for JS ----

    pub fn field_ptr(&self) -> *const f32 {
        self.field.as_ptr()
    }

    pub fn field_len(&self) -> usize {
        self.field.len()
    }

    pub fn n(&self) -> usize {
        self.n
    }

    // ---- Core: advance by tau using stable substeps ----

    pub fn step_tau(&mut self) -> StepInfo {
        // dt_max = dx^2 / (4 alpha)
        let dt_max = (self.dx * self.dx) / (4.0 * self.alpha);
        let dt = self.s_run * dt_max;

        let tau = self.get_tau();
        let mut k = (tau / dt).ceil() as u32;
        if k < 1 {
            k = 1;
        }
        let dt_prime = tau / (k as f32); // <= dt

        let t0 = now_ms();

        for _ in 0..k {
            self.explicit_step(dt_prime);
        }

        let t1 = now_ms();
        StepInfo {
            k,
            compute_ms: t1 - t0,
            tau,
        }
    }

    // ---- Internal numeric routines ----

    fn explicit_step(&mut self, dt: f32) {
        let n = self.n;
        let dx2 = self.dx * self.dx;
        let c = self.alpha * dt / dx2;

        // interior update only
        for y in 1..(n - 1) {
            let row = y * n;
            for x in 1..(n - 1) {
                let i = row + x;

                let u = self.field[i];
                let up = self.field[i - n];
                let down = self.field[i + n];
                let left = self.field[i - 1];
                let right = self.field[i + 1];

                // u_new = u + alpha*dt*(laplacian)
                // laplacian = (up+down+left+right - 4u)/dx^2
                let lap = (up + down + left + right) - 4.0 * u;
                self.next[i] = (u + c * lap).clamp(0.0, 1.0);
            }
        }

        // boundaries -> 0
        self.swap_buffers();
        self.apply_dirichlet_bc();
    }

    fn apply_dirichlet_bc(&mut self) {
        let n = self.n;
        // top and bottom rows
        for x in 0..n {
            self.field[x] = 0.0;
            self.field[(n - 1) * n + x] = 0.0;
        }
        // left and right cols
        for y in 0..n {
            self.field[y * n] = 0.0;
            self.field[y * n + (n - 1)] = 0.0;
        }
    }

    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.field, &mut self.next);
        // keep next clean-ish (optional)
        self.next.fill(0.0);
    }
}

#[wasm_bindgen]
pub struct StepInfo {
    pub k: u32,
    pub compute_ms: f64,
    pub tau: f32,
}

#[wasm_bindgen]
impl StepInfo {
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> u32 { self.k }

    #[wasm_bindgen(getter)]
    pub fn compute_ms(&self) -> f64 { self.compute_ms }

    #[wasm_bindgen(getter)]
    pub fn tau(&self) -> f32 { self.tau }
}

// High-resolution timer for browsers
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}
