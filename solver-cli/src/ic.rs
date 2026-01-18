use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub enum IcType {
    Gaussians,
    Rectangles,
    SmoothNoise,
    GradientMix,
}

impl IcType {
    pub fn as_str(&self) -> &'static str {
        match self {
            IcType::Gaussians => "gaussians",
            IcType::Rectangles => "rectangles",
            IcType::SmoothNoise => "smooth_noise",
            IcType::GradientMix => "gradient_mix",
        }
    }
}

pub fn sample_ic_type<R: Rng>(rng: &mut R) -> IcType {
    // You can adjust proportions later.
    // Keep it simple and fairly uniform for v1.
    match rng.gen_range(0..4) {
        0 => IcType::Gaussians,
        1 => IcType::Rectangles,
        2 => IcType::SmoothNoise,
        _ => IcType::GradientMix,
    }
}

/// Fill a field (len = n*n) with an IC in [0,1]. Boundary is left for the solver to clamp.
pub fn generate_ic<R: Rng>(rng: &mut R, n: usize, ic: IcType) -> Vec<f32> {
    let mut f = vec![0.0f32; n * n];

    match ic {
        IcType::Gaussians => {
            let blobs = rng.gen_range(1..=3);
            for _ in 0..blobs {
                let cx = rng.gen_range(0.15..0.85) * (n as f32 - 1.0);
                let cy = rng.gen_range(0.15..0.85) * (n as f32 - 1.0);
                let sigma = rng.gen_range(1.5..6.0);
                let amp = rng.gen_range(0.6..1.0);

                for y in 0..n {
                    for x in 0..n {
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let r2 = dx * dx + dy * dy;
                        let v = amp * (-0.5 * r2 / (sigma * sigma)).exp();
                        f[y * n + x] += v;
                    }
                }
            }
        }

        IcType::Rectangles => {
            let rects = rng.gen_range(1..=4);
            for _ in 0..rects {
                let x0 = rng.gen_range(1..n / 2);
                let y0 = rng.gen_range(1..n / 2);
                let w = rng.gen_range(2..n / 2);
                let h = rng.gen_range(2..n / 2);
                let val = rng.gen_range(0.5..1.0);

                let x1 = (x0 + w).min(n - 2);
                let y1 = (y0 + h).min(n - 2);

                for y in y0..=y1 {
                    for x in x0..=x1 {
                        f[y * n + x] = f[y * n + x].max(val);
                    }
                }
            }
        }

        IcType::SmoothNoise => {
            // start with white noise
            for i in 0..f.len() {
                f[i] = rng.gen_range(0.0..1.0);
            }
            // cheap smoothing: 2 passes of box blur
            f = box_blur(&f, n, 2);
        }

        IcType::GradientMix => {
            // gradient + 1 gaussian
            let dir = rng.gen_range(0..4);
            for y in 0..n {
                for x in 0..n {
                    let t = match dir {
                        0 => x as f32 / (n as f32 - 1.0),
                        1 => y as f32 / (n as f32 - 1.0),
                        2 => 1.0 - (x as f32 / (n as f32 - 1.0)),
                        _ => 1.0 - (y as f32 / (n as f32 - 1.0)),
                    };
                    f[y * n + x] = 0.6 * t;
                }
            }
            // add one gaussian bump
            let cx = rng.gen_range(0.2..0.8) * (n as f32 - 1.0);
            let cy = rng.gen_range(0.2..0.8) * (n as f32 - 1.0);
            let sigma = rng.gen_range(2.0..7.0);
            let amp = rng.gen_range(0.4..0.9);

            for y in 0..n {
                for x in 0..n {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r2 = dx * dx + dy * dy;
                    f[y * n + x] += amp * (-0.5 * r2 / (sigma * sigma)).exp();
                }
            }
        }
    }

    // normalize to [0,1]
    normalize_01(&mut f);
    f
}

fn normalize_01(f: &mut [f32]) {
    let mut mx = 0.0f32;
    for &v in f.iter() {
        if v > mx { mx = v; }
    }
    if mx > 0.0 {
        for v in f.iter_mut() {
            *v = (*v / mx).clamp(0.0, 1.0);
        }
    }
}

fn box_blur(src: &[f32], n: usize, passes: usize) -> Vec<f32> {
    let mut cur = src.to_vec();
    let mut tmp = vec![0.0f32; n * n];

    for _ in 0..passes {
        for y in 0..n {
            for x in 0..n {
                let mut sum = 0.0;
                let mut cnt = 0.0;
                for yy in y.saturating_sub(1)..=(y + 1).min(n - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(n - 1) {
                        sum += cur[yy * n + xx];
                        cnt += 1.0;
                    }
                }
                tmp[y * n + x] = sum / cnt;
            }
        }
        std::mem::swap(&mut cur, &mut tmp);
    }
    cur
}
