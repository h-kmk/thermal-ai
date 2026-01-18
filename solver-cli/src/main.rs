mod ic;

use clap::Parser;
use ic::{generate_ic, sample_ic_type};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
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

    /// Split label written into meta.jsonl (train|val|test|ood)
    #[arg(long, default_value = "train")]
    split: String,

    /// Grid size N (NxN)
    #[arg(long, default_value_t = 64)]
    n: usize,

    /// Trajectory start index (for deterministic split-by-range)
    #[arg(long, default_value_t = 0)]
    traj_start: usize,

    /// Number of trajectories to generate
    #[arg(long, default_value_t = 100)]
    traj_count: usize,

    /// How many (u^t -> u^{t+tau}) samples per trajectory
    #[arg(long, default_value_t = 8)]
    t_steps: usize,

    /// Diffusivity alpha min (sampled per-trajectory)
    #[arg(long, default_value_t = 0.05)]
    alpha_min: f32,

    /// Diffusivity alpha max (sampled per-trajectory)
    #[arg(long, default_value_t = 0.5)]
    alpha_max: f32,

    /// Comma-separated mu set, e.g. "2,5,10,20"
    #[arg(long, default_value = "2,5,10,20")]
    mu_set: String,

    /// Reference safety factor s_ref (labels)
    #[arg(long, default_value_t = 0.4)]
    s_ref: f32,

    /// Base RNG seed (reproducibility)
    #[arg(long, default_value_t = 123)]
    seed: u64,
}

#[derive(Serialize)]
struct MetaRow {
    global_sample_idx: u64,
    split: String,

    traj_idx: usize,     // global traj index (traj_start + local)
    step_idx: usize,

    base_seed: u64,
    traj_seed: u64,

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

    if args.alpha_max <= args.alpha_min {
        return Err("alpha_max must be > alpha_min".into());
    }

    let mu_values = parse_mu_set(&args.mu_set)?;
    if mu_values.is_empty() {
        return Err("mu_set parsed to empty set".into());
    }

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

    for local_traj in 0..args.traj_count {
        let traj_idx = args.traj_start + local_traj;

        // Deterministic per-trajectory seed
        // (stable split-by-range; reproducible even if you regenerate)
        let traj_seed = args.seed ^ ((traj_idx as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let mut rng = ChaCha8Rng::seed_from_u64(traj_seed);

        // Sample alpha per trajectory (cleaner than per-step)
        let alpha = rng.gen_range(args.alpha_min..args.alpha_max);

        // Sample IC type + generate IC field
        let ic_t = sample_ic_type(&mut rng);
        let ic_field = generate_ic(&mut rng, args.n, ic_t);

        // Create solver
        let mut s = SolverCore::new(args.n).map_err(|e| format!("SolverCore::new: {e}"))?;
        s.set_alpha(alpha);
        s.set_s_ref(args.s_ref);

        // Apply IC into solver
        for y in 0..args.n {
            for x in 0..args.n {
                s.set_cell(x, y, ic_field[y * args.n + x]);
            }
        }
        s.finalize_ic();

        let dx_val = s.get_dx();

        // Roll forward and collect pairs
        for step_idx in 0..args.t_steps {
            // Sample mu per step
            let mu = *mu_values.choose(&mut rng).unwrap();
            s.set_mu(mu);

            // Input
            let u_in = s.clone_field();

            // Advance by tau using reference stepping
            let (k_used, tau_val) = s.step_tau_ref();

            // Target
            let u_out = s.clone_field();

            // Write binaries
            write_f32_vec(&mut input_writer, &u_in)?;
            write_f32_vec(&mut target_writer, &u_out)?;

            // Write metadata row (JSONL)
            let row = MetaRow {
                global_sample_idx: global_idx,
                split: args.split.clone(),

                traj_idx,
                step_idx,

                base_seed: args.seed,
                traj_seed,

                n: args.n,
                dx: dx_val,

                alpha,
                mu,
                tau: tau_val,

                s_ref: args.s_ref,
                k_used_ref: k_used,

                ic_type: ic_t.as_str().to_string(),
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
        "Samples: {} (traj_count={} * t_steps={})",
        global_idx, args.traj_count, args.t_steps
    );

    Ok(())
}

fn write_f32_vec<W: Write>(w: &mut W, v: &[f32]) -> std::io::Result<()> {
    for &x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn parse_mu_set(s: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let v: f32 = p.parse()?;
        if v < 0.0 {
            return Err("mu_set cannot contain negative values".into());
        }
        out.push(v);
    }
    // remove duplicates (stable order)
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out.dedup();
    Ok(out)
}
