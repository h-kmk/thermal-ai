pub struct SolverCore {
    n: usize,
    alpha: f32,
    mu: f32,
    s_run: f32,
    s_ref: f32,
    dx: f32,
    field: Vec<f32>,
    next: Vec<f32>,
}

impl SolverCore {
    pub fn new(n: usize) -> Result<SolverCore, String> {
        if n < 3 {
            return Err("n must be >= 3".into());
        }
        let dx = 1.0f32 / ((n - 1) as f32);
        let size = n * n;

        Ok(SolverCore {
            n,
            alpha: 0.2,
            mu: 10.0,
            s_run: 0.8,
            s_ref: 0.35,
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
        self.s_run = s.clamp(0.05, 0.99);
    }

    pub fn get_dx(&self) -> f32 {
        self.dx
    }

    pub fn get_tau(&self) -> f32 {
        (self.mu * self.dx * self.dx) / self.alpha
    }

    pub fn clear(&mut self) {
        self.field.fill(0.0);
        self.next.fill(0.0);
    }

    pub fn add_hotspot(&mut self, x: usize, y: usize, value: f32) {
        if x >= self.n || y >= self.n {
            return;
        }
        let idx = y * self.n + x;
        self.field[idx] = (self.field[idx] + value).clamp(0.0, 1.0);
        self.apply_dirichlet_bc();
    }
    pub fn set_s_ref(&mut self, s: f32) {
        self.s_ref = s.clamp(0.05, 0.99);
    }
    
    pub fn get_s_run(&self) -> f32 {
        self.s_run
    }
    
    pub fn get_s_ref(&self) -> f32 {
        self.s_ref
    }
    

    // ---- Accessors ----
    pub fn n(&self) -> usize {
        self.n
    }

    pub fn field(&self) -> &[f32] {
        &self.field
    }

    // ---- Core: advance by tau using stable substeps ----
    // Returns: (k_used, tau)

    fn step_tau_with_s(&mut self, s: f32) -> (u32, f32) {
        let dt_max = (self.dx * self.dx) / (4.0 * self.alpha);
        let dt = s * dt_max;
    
        let tau = self.get_tau();
        let mut k = (tau / dt).ceil() as u32;
        if k < 1 { k = 1; }
    
        let dt_prime = tau / (k as f32);
    
        for _ in 0..k {
            self.explicit_step(dt_prime);
        }
    
        (k, tau)
    }
    
    pub fn step_tau_run(&mut self) -> (u32, f32) {
        self.step_tau_with_s(self.s_run)
    }
    
    pub fn step_tau_ref(&mut self) -> (u32, f32) {
        self.step_tau_with_s(self.s_ref)
    }

    pub fn set_cell(&mut self, x: usize, y: usize, value: f32) {
        if x >= self.n || y >= self.n { return; }
        let idx = y * self.n + x;
        self.field[idx] = value.clamp(0.0, 1.0);
    }
    pub fn finalize_ic(&mut self) {
        self.apply_dirichlet_bc();
    }
    pub fn clone_field(&self) -> Vec<f32> {
        self.field.clone()
    }
    
    
    
        

    // ---- Internal numeric routines ----
    fn explicit_step(&mut self, dt: f32) {
        let n = self.n;
        let dx2 = self.dx * self.dx;
        let c = self.alpha * dt / dx2;

        for y in 1..(n - 1) {
            let row = y * n;
            for x in 1..(n - 1) {
                let i = row + x;

                let u = self.field[i];
                let up = self.field[i - n];
                let down = self.field[i + n];
                let left = self.field[i - 1];
                let right = self.field[i + 1];

                let lap = (up + down + left + right) - 4.0 * u;
                self.next[i] = (u + c * lap).clamp(0.0, 1.0);
            }
        }

        self.swap_buffers();
        self.apply_dirichlet_bc();
    }

    fn apply_dirichlet_bc(&mut self) {
        let n = self.n;
        for x in 0..n {
            self.field[x] = 0.0;
            self.field[(n - 1) * n + x] = 0.0;
        }
        for y in 0..n {
            self.field[y * n] = 0.0;
            self.field[y * n + (n - 1)] = 0.0;
        }
    }

    fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.field, &mut self.next);
        self.next.fill(0.0);
    }
}
