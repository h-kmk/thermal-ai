# Dataset Format (Thermal-AI v1)

Each sample is a pair (u_in -> u_out) on an N x N grid of float32 values.

Files:
- input.bin: concatenated float32 fields for u^t
- target.bin: concatenated float32 fields for u^{t+tau}
- meta.jsonl: one JSON object per sample (one line per sample)

Binary layout:
- Little-endian float32
- One sample = N*N float32 values
- Sample i occupies bytes:
  - [i * (N*N*4), (i+1) * (N*N*4))

Meta schema (minimum fields):
- global_sample_idx (u64)
- split (train|val|test|ood)
- traj_idx, step_idx
- traj_seed (u64)
- n (usize), dx (f32)
- alpha (f32)
- mu (f32)
- tau (f32)
- s_ref (f32)
- k_used_ref (u32)
- ic_type (string)