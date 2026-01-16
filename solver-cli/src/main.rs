use clap::Parser;
use serde::Serialize;
use solver_core::SolverCore;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Output directory
    #[arg(long)]
    out: PathBuf,

    /// Grid size N (NxN)
    #[arg(long, default_value_t = 64)]
    n: usize,

    /// Number of trajectories (scenarios)
    #[arg(long, default_value_t = 100)]
    count: usize,

    /// How many (u^t -> u^{t+tau}) samples per trajectory
    #[arg(long, default_value_t = 8)]
    t_steps: usize,

    /// Diffusivity alpha
    #[arg(long, default_value_t = 0.2)]
    alpha: f32,

    /// Jump size mu
    #[arg(long, default_value_t = 10.0)]
    mu: f32,

    /// Reference safety factor s_ref
    #[arg(long, default_value_t = 0.35)]
    s_ref: f32,

    /// RNG seed (reproducibility)
    #[arg(long, default_value_t = 123)]
    seed: u64,
}

#[derive(Serialize)]
struct MetaRow {
    global_sample_idx: u64,
    traj_idx: usize,
    step_idx: usize,

    seed: u64,
    n: usize,
    dx: f32,

    alpha: f32,
    mu: f32,
    tau: f32,

    s_ref: f32,
    k_used_ref: u32,

    ic_type: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    fs::create_dir_all(&args.out)?;

    let mut input_writer = BufWriter::new(File::create(args.out.join("input.bin"))?);
    let mut target_writer = BufWriter::new(File::create(args.out.join("target.bin"))?);

    let mut meta_file = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(args.out.join("meta.jsonl"))?,
    );

    let mut global_idx: u64 = 0;

    for traj_idx in 0..args.count {
        // --- Create solver ---
        let mut s = SolverCore::new(args.n).map_err(|e| format!("SolverCore::new: {e}"))?;
        s.set_alpha(args.alpha);
        s.set_mu(args.mu);
        s.set_s_ref(args.s_ref);

        // --- Deterministic IC (simple for v0): one centered gaussian-like blob approximation ---
        // We'll make a small filled square stamp around center. This is enough to validate the pipeline.
        let cx = args.n / 2;
        let cy = args.n / 2;
        let r = 2usize;
        for yy in cy.saturating_sub(r)..=(cy + r).min(args.n - 1) {
            for xx in cx.saturating_sub(r)..=(cx + r).min(args.n - 1) {
                // stronger in center
                let dx = (xx as i32 - cx as i32).abs() as f32;
                let dy = (yy as i32 - cy as i32).abs() as f32;
                let dist2 = dx * dx + dy * dy;
                let v = (-0.5 * dist2).exp(); // in (0,1]
                s.set_cell(xx, yy, v);
            }
        }
        s.finalize_ic();

        let dx_val = s.get_dx();
        let tau_val = s.get_tau();

        // --- Roll forward and collect pairs ---
        for step_idx in 0..args.t_steps {
            // Input
            let u_in = s.clone_field();

            // Advance by tau using reference stepping
            let (k_used, _tau) = s.step_tau_ref();

            // Target
            let u_out = s.clone_field();

            // Write binaries (little-endian f32)
            write_f32_vec(&mut input_writer, &u_in)?;
            write_f32_vec(&mut target_writer, &u_out)?;

            // Write metadata row (JSONL)
            let row = MetaRow {
                global_sample_idx: global_idx,
                traj_idx,
                step_idx,
                seed: args.seed, // (v0) single seed, later: seed per traj
                n: args.n,
                dx: dx_val,
                alpha: args.alpha,
                mu: args.mu,
                tau: tau_val,
                s_ref: args.s_ref,
                k_used_ref: k_used,
                ic_type: "center_stamp_v0".to_string(),
            };

            serde_json::to_writer(&mut meta_file, &row)?;
            meta_file.write_all(b"\n")?;

            global_idx += 1;
        }
    }

    input_writer.flush()?;
    target_writer.flush()?;
    meta_file.flush()?;

    println!("Wrote dataset to: {}", args.out.display());
    println!(
        "Samples: {} (count={} * t_steps={})",
        global_idx, args.count, args.t_steps
    );

    Ok(())
}

fn write_f32_vec<W: Write>(w: &mut W, v: &[f32]) -> std::io::Result<()> {
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}
