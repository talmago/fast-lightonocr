//! Isolated KV-cache representation microbenchmark.
//!
//! Measures host-side update strategies without ONNX Runtime. Shapes match
//! LightOnOCR-2-1B: 28 layers, 8 KV heads, head_dim 128, batch 1.
//!
//! Run:
//! ```text
//! cargo run --release --example kv_cache_bench -- 256
//! ```

use std::time::{Duration, Instant};

const LAYERS: usize = 28;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BATCH: usize = 1;
const WARMUP: usize = 2;
const ITERS: usize = 5;

fn values_per_tensor(seq: usize) -> usize {
    BATCH * HEADS * seq * HEAD_DIM
}

fn bytes(values: usize) -> u64 {
    (values * std::mem::size_of::<f32>()) as u64
}

fn present_token_slice(present: &[f32], seq: usize, token: usize) -> Vec<f32> {
    // Layout (B,H,S,D): for each head, slice [token*D .. (token+1)*D]
    let mut out = Vec::with_capacity(BATCH * HEADS * HEAD_DIM);
    for head in 0..HEADS {
        let base = head * seq * HEAD_DIM + token * HEAD_DIM;
        out.extend_from_slice(&present[base..base + HEAD_DIM]);
    }
    out
}

fn mock_present(seq: usize, fill: f32) -> Vec<f32> {
    vec![fill; values_per_tensor(seq)]
}

#[derive(Debug, Clone, Copy)]
struct BenchResult {
    name: &'static str,
    wall: Duration,
    bytes_touched: u64,
    ort_legal_bhsd: bool,
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn run_strategy(
    name: &'static str,
    max_seq: usize,
    ort_legal_bhsd: bool,
    mut step: impl FnMut(usize, &[Vec<f32>]) -> u64,
) -> BenchResult {
    // Pre-generate mock present tensors for each step length.
    let presents: Vec<Vec<f32>> = (1..=max_seq)
        .map(|seq| mock_present(seq, seq as f32))
        .collect();

    for _ in 0..WARMUP {
        let _ = step(max_seq, &presents);
    }

    let mut samples = Vec::with_capacity(ITERS);
    let mut bytes_touched = 0u64;
    for _ in 0..ITERS {
        let started = Instant::now();
        bytes_touched = step(max_seq, &presents);
        samples.push(started.elapsed());
    }

    BenchResult {
        name,
        wall: median(samples),
        bytes_touched,
        ort_legal_bhsd,
    }
}

/// Current production path: allocate a fresh Vec per present tensor each step.
fn full_copy_replace(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    let mut cache: Vec<(Vec<f32>, Vec<f32>)> =
        (0..LAYERS).map(|_| (Vec::new(), Vec::new())).collect();
    let mut bytes_touched = 0u64;

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let mut next = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            let key = present.clone();
            let value = present.clone();
            bytes_touched += bytes(key.len()) + bytes(value.len());
            next.push((key, value));
        }
        cache = next;
    }

    // Keep cache live so optimizer cannot elide work.
    std::hint::black_box(&cache);
    bytes_touched
}

/// Preallocate to max_seq and overwrite growing prefixes in place.
fn reusable_buffers(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    let capacity = values_per_tensor(max_seq);
    let mut cache: Vec<(Vec<f32>, Vec<f32>)> = (0..LAYERS)
        .map(|_| (vec![0.0; capacity], vec![0.0; capacity]))
        .collect();
    let mut bytes_touched = 0u64;

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let len = values_per_tensor(seq);
        for (key, value) in &mut cache {
            key[..len].copy_from_slice(&present[..len]);
            value[..len].copy_from_slice(&present[..len]);
            bytes_touched += bytes(len) * 2;
        }
    }

    std::hint::black_box(&cache);
    bytes_touched
}

/// Double buffer: write into back buffer, swap.
fn double_buffer(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    let capacity = values_per_tensor(max_seq);
    let mut front: Vec<(Vec<f32>, Vec<f32>)> = (0..LAYERS)
        .map(|_| (vec![0.0; capacity], vec![0.0; capacity]))
        .collect();
    let mut back: Vec<(Vec<f32>, Vec<f32>)> = (0..LAYERS)
        .map(|_| (vec![0.0; capacity], vec![0.0; capacity]))
        .collect();
    let mut bytes_touched = 0u64;

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let len = values_per_tensor(seq);
        for (key, value) in &mut back {
            key[..len].copy_from_slice(&present[..len]);
            value[..len].copy_from_slice(&present[..len]);
            bytes_touched += bytes(len) * 2;
        }
        std::mem::swap(&mut front, &mut back);
    }

    std::hint::black_box(&front);
    bytes_touched
}

/// Naive append into (B,H,S,D): strided insert of one token per head.
fn append_inplace_bhsd(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    let mut cache: Vec<(Vec<f32>, Vec<f32>)> =
        (0..LAYERS).map(|_| (Vec::new(), Vec::new())).collect();
    let mut bytes_touched = 0u64;

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let token = present_token_slice(present, seq, seq - 1);
        for (key, value) in &mut cache {
            // Rebuild (B,H,S,D) by inserting into each head slab.
            // Equivalent to growing S by 1 with strided memmoves.
            let old_seq = seq - 1;
            if old_seq == 0 {
                *key = token.clone();
                *value = token.clone();
                bytes_touched += bytes(token.len()) * 2;
                continue;
            }

            let mut new_key = Vec::with_capacity(values_per_tensor(seq));
            let mut new_value = Vec::with_capacity(values_per_tensor(seq));
            for head in 0..HEADS {
                let old_base = head * old_seq * HEAD_DIM;
                let tok_base = head * HEAD_DIM;
                new_key.extend_from_slice(&key[old_base..old_base + old_seq * HEAD_DIM]);
                new_key.extend_from_slice(&token[tok_base..tok_base + HEAD_DIM]);
                new_value.extend_from_slice(&value[old_base..old_base + old_seq * HEAD_DIM]);
                new_value.extend_from_slice(&token[tok_base..tok_base + HEAD_DIM]);
            }
            bytes_touched += bytes(new_key.len()) + bytes(new_value.len());
            *key = new_key;
            *value = new_value;
        }
    }

    std::hint::black_box(&cache);
    bytes_touched
}

/// Append-friendly layout (B,H,D,S): extend contiguous per-head rows, then
/// transpose to ORT-legal (B,H,S,D) each step.
fn append_alt_layout(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    // Store as (H, D, S) flat: index [head][dim][seq]
    let mut cache: Vec<(Vec<f32>, Vec<f32>)> = (0..LAYERS)
        .map(|_| {
            (
                Vec::with_capacity(HEADS * HEAD_DIM * max_seq),
                Vec::with_capacity(HEADS * HEAD_DIM * max_seq),
            )
        })
        .collect();
    let mut bytes_touched = 0u64;
    let mut ort_views: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let token = present_token_slice(present, seq, seq - 1);
        ort_views.clear();

        for (key, value) in &mut cache {
            // Append token into (H,D,S) storage.
            if seq == 1 {
                // Initialize empty (H,D,0) then push S=1.
                key.clear();
                value.clear();
                key.resize(HEADS * HEAD_DIM, 0.0);
                value.resize(HEADS * HEAD_DIM, 0.0);
                for head in 0..HEADS {
                    for dim in 0..HEAD_DIM {
                        let idx = head * HEAD_DIM + dim;
                        key[idx] = token[head * HEAD_DIM + dim];
                        value[idx] = token[head * HEAD_DIM + dim];
                    }
                }
                bytes_touched += bytes(HEADS * HEAD_DIM) * 2;
            } else {
                // Grow S dimension: for each (head,dim), push one value.
                // Represented as H slabs of D*S.
                let mut new_key = Vec::with_capacity(HEADS * HEAD_DIM * seq);
                let mut new_value = Vec::with_capacity(HEADS * HEAD_DIM * seq);
                let old_seq = seq - 1;
                for head in 0..HEADS {
                    for dim in 0..HEAD_DIM {
                        let old_base = (head * HEAD_DIM + dim) * old_seq;
                        new_key.extend_from_slice(&key[old_base..old_base + old_seq]);
                        new_key.push(token[head * HEAD_DIM + dim]);
                        new_value.extend_from_slice(&value[old_base..old_base + old_seq]);
                        new_value.push(token[head * HEAD_DIM + dim]);
                    }
                }
                bytes_touched += bytes(new_key.len()) + bytes(new_value.len());
                *key = new_key;
                *value = new_value;
            }

            // Transpose (H,D,S) -> (H,S,D) for ORT.
            let mut ort_key = vec![0.0; values_per_tensor(seq)];
            let mut ort_value = vec![0.0; values_per_tensor(seq)];
            for head in 0..HEADS {
                for s in 0..seq {
                    for dim in 0..HEAD_DIM {
                        let src = (head * HEAD_DIM + dim) * seq + s;
                        let dst = head * seq * HEAD_DIM + s * HEAD_DIM + dim;
                        ort_key[dst] = key[src];
                        ort_value[dst] = value[src];
                    }
                }
            }
            bytes_touched += bytes(ort_key.len()) + bytes(ort_value.len());
            ort_views.push((ort_key, ort_value));
        }
    }

    std::hint::black_box(&ort_views);
    bytes_touched
}

/// Keep reusable (B,H,S,D) buffers; copy only the newest token slice from each
/// full present tensor (delta extract).
fn delta_extract_only(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    let capacity = values_per_tensor(max_seq);
    let mut cache: Vec<(Vec<f32>, Vec<f32>)> = (0..LAYERS)
        .map(|_| (vec![0.0; capacity], vec![0.0; capacity]))
        .collect();
    let mut bytes_touched = 0u64;

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let token = present_token_slice(present, seq, seq - 1);
        let token_bytes = bytes(token.len());
        // Reading the delta still "touches" the source token slice; count write.
        for (key, value) in &mut cache {
            for head in 0..HEADS {
                let dst = head * max_seq * HEAD_DIM + (seq - 1) * HEAD_DIM;
                // Pack densely in growing prefix for ORT view of length seq:
                // use compacted layout key[head*seq*D + (seq-1)*D]
                let dst_compact = head * seq * HEAD_DIM + (seq - 1) * HEAD_DIM;
                let src = head * HEAD_DIM;
                key[dst_compact..dst_compact + HEAD_DIM]
                    .copy_from_slice(&token[src..src + HEAD_DIM]);
                value[dst_compact..dst_compact + HEAD_DIM]
                    .copy_from_slice(&token[src..src + HEAD_DIM]);
                let _ = dst; // reserved for padded variant
            }
            bytes_touched += token_bytes * 2;
        }
    }

    std::hint::black_box(&cache);
    bytes_touched
}

/// Fixed max-length buffers; always copy full present into prefix (simulates
/// padded cache host update without growing allocations).
fn padded_max_len(max_seq: usize, presents: &[Vec<f32>]) -> u64 {
    let capacity = values_per_tensor(max_seq);
    let mut cache: Vec<(Vec<f32>, Vec<f32>)> = (0..LAYERS)
        .map(|_| (vec![0.0; capacity], vec![0.0; capacity]))
        .collect();
    let mut bytes_touched = 0u64;

    for seq in 1..=max_seq {
        let present = &presents[seq - 1];
        let len = values_per_tensor(seq);
        for (key, value) in &mut cache {
            // Write into fixed buffer; ORT would see shape S=max_seq with mask.
            // Host update still only needs the used prefix here.
            key[..len].copy_from_slice(&present[..len]);
            value[..len].copy_from_slice(&present[..len]);
            bytes_touched += bytes(len) * 2;
        }
    }

    std::hint::black_box(&cache);
    bytes_touched
}

fn print_result(result: &BenchResult, baseline: Duration) {
    let speedup = if result.wall > Duration::ZERO {
        baseline.as_secs_f64() / result.wall.as_secs_f64()
    } else {
        f64::INFINITY
    };
    println!(
        "{:<28} {:>10.3} ms  bytes={:>12}  ort_legal={}  vs_full_copy={:.2}x",
        result.name,
        result.wall.as_secs_f64() * 1000.0,
        result.bytes_touched,
        result.ort_legal_bhsd,
        speedup
    );
}

fn main() {
    let max_seq = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(256usize);

    println!("KV cache microbench");
    println!("  layers={LAYERS} heads={HEADS} head_dim={HEAD_DIM} max_seq={max_seq}");
    println!("  warmup={WARMUP} iters={ITERS} (median)");
    println!();

    let results = vec![
        run_strategy("FullCopyReplace", max_seq, true, full_copy_replace),
        run_strategy("ReusableBuffers", max_seq, true, reusable_buffers),
        run_strategy("DoubleBuffer", max_seq, true, double_buffer),
        run_strategy("AppendInPlaceBHSD", max_seq, true, append_inplace_bhsd),
        run_strategy("AppendAltLayout", max_seq, true, append_alt_layout),
        run_strategy("DeltaExtractOnly", max_seq, true, delta_extract_only),
        run_strategy("PaddedMaxLen", max_seq, true, padded_max_len),
    ];

    let baseline = results[0].wall;
    for result in &results {
        print_result(result, baseline);
    }
}
