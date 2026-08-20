use crate::sim::{CpuStepMode, SimState};
use rand::Rng;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolvedStepMode {
    Naive,
    GridExact,
}

#[derive(Clone, Copy, Debug, Default)]
struct StepStats {
    used_grid: bool,
    neighbor_checks: u64,
    grid_res: usize,
}

/// CPU stepping:
/// - Naive: exact O(N²)
/// - GridExact: exact neighbor cull using a uniform grid
///
/// GridExact is BIT-IDENTICAL to Naive: candidate neighbors are gathered,
/// sorted by index, and accumulated in ascending order — the same f32 sum
/// Naive produces (out-of-range pairs contribute exactly nothing). This
/// matters because the period-2 oscillating "objects" this project studies
/// live on a floating-point knife edge; switching step modes must not
/// perturb them.
pub fn cpu_step(state: &mut SimState) {
    let frame_start = Instant::now();

    let count = state.particles.len();
    if count == 0 {
        state.last_step_used_grid = false;
        state.last_neighbor_checks = 0;
        state.last_grid_res = 1;
        state.last_step_ms = 0.0;
        return;
    }

    let mode = resolve_step_mode(state, count);
    let stats = match mode {
        ResolvedStepMode::Naive => cpu_step_naive(state),
        ResolvedStepMode::GridExact => cpu_step_grid_exact(state),
    };

    // Apply reactions after physics integration
    apply_reactions(state);

    state.last_step_used_grid = stats.used_grid;
    state.last_neighbor_checks = stats.neighbor_checks;
    state.last_grid_res = stats.grid_res;

    let elapsed_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
    state.last_step_ms = elapsed_ms;

    if state.avg_step_ms == 0.0 {
        state.avg_step_ms = elapsed_ms;
    } else {
        state.avg_step_ms = state.avg_step_ms * 0.90 + elapsed_ms * 0.10;
    }
}

fn resolve_step_mode(state: &SimState, count: usize) -> ResolvedStepMode {
    match state.params.cpu_step_mode {
        CpuStepMode::Naive => ResolvedStepMode::Naive,
        CpuStepMode::GridExact => ResolvedStepMode::GridExact,
        CpuStepMode::Auto => {
            if count >= state.params.auto_grid_threshold
                && state.params.r_max > 0.0
                && state.params.bounds > 0.0
            {
                ResolvedStepMode::GridExact
            } else {
                ResolvedStepMode::Naive
            }
        }
    }
}

fn cpu_step_naive(state: &mut SimState) -> StepStats {
    let count = state.particles.len();
    if count == 0 {
        return StepStats::default();
    }

    let dt = state.params.dt;
    let r_max = state.params.scaled_r_max();
    let r_max_sq = r_max * r_max;
    let force_scale = state.params.force_scale;
    let friction = state.params.friction;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;
    let beta = state.params.beta;
    let max_speed = state.params.scaled_max_speed();
    let type_count = state.params.type_count;

    let damping = friction.powf(dt * 60.0);

    // Prepare scratch buffer
    state.vel_scratch.resize(count, [0.0; 3]);
    for (i, p) in state.particles.iter().enumerate() {
        state.vel_scratch[i] = p.velocity;
    }
    let mut neighbor_checks = 0_u64;

    // Compute forces
    for i in 0..count {
        let pi_pos = state.particles[i].position;
        let pi_kind = state.particles[i].kind as usize;

        let mut fx = 0.0_f32;
        let mut fy = 0.0_f32;
        let mut fz = 0.0_f32;

        for j in 0..count {
            neighbor_checks += 1;
            if i == j {
                continue;
            }

            let pj_pos = state.particles[j].position;
            let pj_kind = state.particles[j].kind as usize;

            let mut dx = pj_pos[0] - pi_pos[0];
            let mut dy = pj_pos[1] - pi_pos[1];
            let mut dz = pj_pos[2] - pi_pos[2];

            if wrap {
                dx = wrap_delta(dx, bounds);
                dy = wrap_delta(dy, bounds);
                dz = wrap_delta(dz, bounds);
            }

            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < 1e-10 || dist_sq > r_max_sq {
                continue;
            }

            let dist = dist_sq.sqrt();
            let dn = dist / r_max;

            let a = if dn < beta {
                dn / beta - 1.0
            } else {
                let matrix_index = pi_kind * type_count + pj_kind;
                let attr = if matrix_index < state.force_matrix.len() {
                    state.force_matrix[matrix_index]
                } else {
                    0.0 // Default to neutral force if out of bounds
                };
                attr * (1.0 - ((2.0 * dn - 1.0 - beta) / (1.0 - beta)).abs())
            };

            let inv_dist = 1.0 / dist;
            fx += a * dx * inv_dist;
            fy += a * dy * inv_dist;
            fz += a * dz * inv_dist;
        }

        let vel = &mut state.vel_scratch[i];
        vel[0] = (vel[0] + fx * force_scale * dt) * damping;
        vel[1] = (vel[1] + fy * force_scale * dt) * damping;
        vel[2] = (vel[2] + fz * force_scale * dt) * damping;

        clamp_velocity(vel, max_speed);
    }

    // Apply velocities to particle positions
    let dt = state.params.dt;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;

    for (i, p) in state.particles.iter_mut().enumerate() {
        p.velocity = state.vel_scratch[i];
        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;

        if wrap {
            for coord in &mut p.position {
                *coord = coord.rem_euclid(bounds);
            }
        }
    }

    StepStats {
        used_grid: false,
        neighbor_checks,
        grid_res: 1,
    }
}

fn cpu_step_grid_exact(state: &mut SimState) -> StepStats {
    let count = state.particles.len();
    if count == 0 {
        return StepStats::default();
    }

    let dt = state.params.dt;
    let r_max = state.params.scaled_r_max();
    let r_max_sq = r_max * r_max;
    let force_scale = state.params.force_scale;
    let friction = state.params.friction;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;
    let beta = state.params.beta;
    let max_speed = state.params.scaled_max_speed();
    let type_count = state.params.type_count;

    let damping = friction.powf(dt * 60.0);

    let grid_res = choose_grid_res(bounds, r_max);
    let total_cells = grid_res * grid_res * grid_res;

    // Prepare buckets scratch buffer
    state.buckets_scratch.resize_with(total_cells, Vec::new);
    for b in &mut state.buckets_scratch {
        b.clear();
    }

    for (i, p) in state.particles.iter().enumerate() {
        let [cx, cy, cz] = cell_coords(p.position, bounds, grid_res, wrap);
        let cell_id = cell_index(cx, cy, cz, grid_res);
        state.buckets_scratch[cell_id].push(i);
    }

    // Prepare velocity scratch buffer
    state.vel_scratch.resize(count, [0.0; 3]);
    for (i, p) in state.particles.iter().enumerate() {
        state.vel_scratch[i] = p.velocity;
    }
    let mut neighbor_checks = 0_u64;

    // Candidate-neighbor scratch, reused across particles.
    let mut neigh: Vec<usize> = Vec::with_capacity(128);

    for i in 0..count {
        let pi_pos = state.particles[i].position;
        let pi_kind = state.particles[i].kind as usize;
        let [cx, cy, cz] = cell_coords(pi_pos, bounds, grid_res, wrap);

        let mut fx = 0.0_f32;
        let mut fy = 0.0_f32;
        let mut fz = 0.0_f32;

        // Gather candidates first, then accumulate in ascending index order.
        // Sorting makes the f32 sum bit-identical to Naive; dedup fixes the
        // double-count when wrap is on and grid_res <= 2 (neighbor offsets
        // alias onto the same cell).
        neigh.clear();
        for ox in -1isize..=1 {
            let Some(nx) = neighbor_axis(cx, ox, grid_res, wrap) else {
                continue;
            };
            for oy in -1isize..=1 {
                let Some(ny) = neighbor_axis(cy, oy, grid_res, wrap) else {
                    continue;
                };
                for oz in -1isize..=1 {
                    let Some(nz) = neighbor_axis(cz, oz, grid_res, wrap) else {
                        continue;
                    };
                    let nid = cell_index(nx, ny, nz, grid_res);
                    neigh.extend_from_slice(&state.buckets_scratch[nid]);
                }
            }
        }
        neigh.sort_unstable();
        neigh.dedup();

        for &j in &neigh {
            neighbor_checks += 1;
            if i == j {
                continue;
            }

            let pj_pos = state.particles[j].position;
            let pj_kind = state.particles[j].kind as usize;

            let mut dx = pj_pos[0] - pi_pos[0];
            let mut dy = pj_pos[1] - pi_pos[1];
            let mut dz = pj_pos[2] - pi_pos[2];

            if wrap {
                dx = wrap_delta(dx, bounds);
                dy = wrap_delta(dy, bounds);
                dz = wrap_delta(dz, bounds);
            }

            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < 1e-10 || dist_sq > r_max_sq {
                continue;
            }

            let dist = dist_sq.sqrt();
            let dn = dist / r_max;

            let a = if dn < beta {
                dn / beta - 1.0
            } else {
                let matrix_index = pi_kind * type_count + pj_kind;
                let attr = if matrix_index < state.force_matrix.len() {
                    state.force_matrix[matrix_index]
                } else {
                    0.0 // Default to neutral force if out of bounds
                };
                attr * (1.0 - ((2.0 * dn - 1.0 - beta) / (1.0 - beta)).abs())
            };

            let inv_dist = 1.0 / dist;
            fx += a * dx * inv_dist;
            fy += a * dy * inv_dist;
            fz += a * dz * inv_dist;
        }

        let vel = &mut state.vel_scratch[i];
        vel[0] = (vel[0] + fx * force_scale * dt) * damping;
        vel[1] = (vel[1] + fy * force_scale * dt) * damping;
        vel[2] = (vel[2] + fz * force_scale * dt) * damping;

        clamp_velocity(vel, max_speed);
    }

    // Apply velocities to particle positions
    let dt = state.params.dt;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;

    for (i, p) in state.particles.iter_mut().enumerate() {
        p.velocity = state.vel_scratch[i];
        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;

        if wrap {
            for coord in &mut p.position {
                *coord = coord.rem_euclid(bounds);
            }
        }
    }

    StepStats {
        used_grid: true,
        neighbor_checks,
        grid_res,
    }
}

fn clamp_velocity(vel: &mut [f32; 3], max_speed: f32) {
    let speed_sq = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
    if speed_sq > max_speed * max_speed {
        let speed = speed_sq.sqrt();
        let scale = max_speed / speed;
        vel[0] *= scale;
        vel[1] *= scale;
        vel[2] *= scale;
    }
}

fn choose_grid_res(bounds: f32, r_max: f32) -> usize {
    if bounds <= 0.0 || r_max <= 0.0 {
        return 1;
    }

    let raw = (bounds / r_max).floor() as usize;
    raw.clamp(1, 200) // cap at 200³ = 8M cells max
}

fn wrap_delta(mut d: f32, bounds: f32) -> f32 {
    let half = bounds * 0.5;
    if d > half {
        d -= bounds;
    } else if d < -half {
        d += bounds;
    }
    d
}

fn cell_coords(pos: [f32; 3], bounds: f32, grid_res: usize, wrap: bool) -> [usize; 3] {
    [
        axis_to_cell(pos[0], bounds, grid_res, wrap),
        axis_to_cell(pos[1], bounds, grid_res, wrap),
        axis_to_cell(pos[2], bounds, grid_res, wrap),
    ]
}

fn axis_to_cell(value: f32, bounds: f32, grid_res: usize, wrap: bool) -> usize {
    if grid_res <= 1 || bounds <= 0.0 {
        return 0;
    }

    let v = if wrap {
        value.rem_euclid(bounds.max(f32::EPSILON))
    } else {
        value.clamp(0.0, (bounds - f32::EPSILON).max(0.0))
    };

    let scaled = (v / bounds) * grid_res as f32;
    scaled.floor().clamp(0.0, (grid_res - 1) as f32) as usize
}

fn cell_index(x: usize, y: usize, z: usize, grid_res: usize) -> usize {
    x * grid_res * grid_res + y * grid_res + z
}

fn neighbor_axis(axis: usize, delta: isize, grid_res: usize, wrap: bool) -> Option<usize> {
    if wrap {
        let g = grid_res as isize;
        Some((axis as isize + delta).rem_euclid(g) as usize)
    } else {
        let v = axis as isize + delta;
        if v < 0 || v >= grid_res as isize {
            None
        } else {
            Some(v as usize)
        }
    }
}

pub fn apply_reactions(state: &mut SimState) {
    if !state.params.reactions_enabled {
        return;
    }

    let mix_r = state.params.scaled_mix_radius();
    if mix_r <= 0.0 {
        return;
    }
    let mix_r_sq = mix_r * mix_r;
    let prob = state.params.reaction_probability;
    let n = state.params.type_count;
    let bounds = state.params.bounds;
    let wrap = state.params.wrap;
    let count = state.particles.len();
    if count == 0 {
        return;
    }

    let mut rng = rand::thread_rng();
    state.reaction_changes_scratch.clear();

    // Build a FRESH grid from POST-integration positions, sized by the
    // reaction radius. Three bugs lived here previously:
    //   1. It reused the force grid built from PRE-integration positions with
    //      LAST frame's grid_res (stats are written after this fn runs), so
    //      fast-moving particles — exactly the ones oscillating "objects" are
    //      made of — were looked up in the wrong cells.
    //   2. Naive mode never cleared the buckets, so after grid mode ran once,
    //      "Naive" reactions kept using an ever-staler grid (and stale indices
    //      could go out of bounds after deletions).
    //   3. Cells were sized by r_max, but mix_radius is independent and can be
    //      larger — the 27-cell neighborhood then misses reaction partners.
    // Now the grid is rebuilt here every call (or skipped for small N), so
    // reaction pairing is exact in every step mode.
    let grid_res = choose_grid_res(bounds, mix_r);
    let use_grid = count >= 1024 && grid_res >= 3;

    if use_grid {
        let total_cells = grid_res * grid_res * grid_res;
        state.buckets_scratch.resize_with(total_cells, Vec::new);
        for b in &mut state.buckets_scratch {
            b.clear();
        }
        for (i, p) in state.particles.iter().enumerate() {
            let [cx, cy, cz] = cell_coords(p.position, bounds, grid_res, wrap);
            state.buckets_scratch[cell_index(cx, cy, cz, grid_res)].push(i);
        }

        for i in 0..count {
            let pi_pos = state.particles[i].position;
            let ri = state.particles[i].kind as usize;
            let [cx, cy, cz] = cell_coords(pi_pos, bounds, grid_res, wrap);

            for ox in -1isize..=1 {
                let Some(nx) = neighbor_axis(cx, ox, grid_res, wrap) else {
                    continue;
                };
                for oy in -1isize..=1 {
                    let Some(ny) = neighbor_axis(cy, oy, grid_res, wrap) else {
                        continue;
                    };
                    for oz in -1isize..=1 {
                        let Some(nz) = neighbor_axis(cz, oz, grid_res, wrap) else {
                            continue;
                        };
                        let nid = cell_index(nx, ny, nz, grid_res);
                        for &j in &state.buckets_scratch[nid] {
                            if j <= i {
                                continue; // each pair once
                            }

                            let pj = &state.particles[j];
                            let mut dx = pj.position[0] - pi_pos[0];
                            let mut dy = pj.position[1] - pi_pos[1];
                            let mut dz = pj.position[2] - pi_pos[2];
                            if wrap {
                                dx = wrap_delta(dx, bounds);
                                dy = wrap_delta(dy, bounds);
                                dz = wrap_delta(dz, bounds);
                            }
                            let dist_sq = dx * dx + dy * dy + dz * dz;
                            if dist_sq > mix_r_sq {
                                continue;
                            }

                            let rj = pj.kind as usize;
                            if ri >= n || rj >= n {
                                continue;
                            }
                            let result_ij = state.reaction_table[ri * n + rj];
                            let result_ji = state.reaction_table[rj * n + ri];
                            if result_ij >= 0 && rng.gen::<f32>() < prob {
                                state.reaction_changes_scratch.push((i, result_ij as u32));
                            }
                            if result_ji >= 0 && rng.gen::<f32>() < prob {
                                state.reaction_changes_scratch.push((j, result_ji as u32));
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Naive pairwise (small N, or grid too coarse to be worthwhile)
        for i in 0..count {
            for j in (i + 1)..count {
                let pi = &state.particles[i];
                let pj = &state.particles[j];
                let mut dx = pj.position[0] - pi.position[0];
                let mut dy = pj.position[1] - pi.position[1];
                let mut dz = pj.position[2] - pi.position[2];
                if wrap {
                    dx = wrap_delta(dx, bounds);
                    dy = wrap_delta(dy, bounds);
                    dz = wrap_delta(dz, bounds);
                }
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq > mix_r_sq {
                    continue;
                }
                let ri = pi.kind as usize;
                let rj = pj.kind as usize;
                if ri >= n || rj >= n {
                    continue;
                }
                let result_ij = state.reaction_table[ri * n + rj];
                let result_ji = state.reaction_table[rj * n + ri];
                if result_ij >= 0 && rng.gen::<f32>() < prob {
                    state.reaction_changes_scratch.push((i, result_ij as u32));
                }
                if result_ji >= 0 && rng.gen::<f32>() < prob {
                    state.reaction_changes_scratch.push((j, result_ji as u32));
                }
            }
        }
    }

    for (idx, new_kind) in state.reaction_changes_scratch.drain(..) {
        if idx < state.particles.len() {
            state.particles[idx].kind = new_kind;
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Particle, SimState};
    use glam::Vec3;

    /// Deterministic pseudo-random f32 in [0, 1) from an integer hash —
    /// no RNG state, so both sim instances get identical particles.
    fn hash01(seed: u32) -> f32 {
        let mut h = seed.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
        h ^= h >> 16;
        h = h.wrapping_mul(0x85EBCA6B);
        h ^= h >> 13;
        (h & 0x00FF_FFFF) as f32 / 16_777_216.0
    }

    fn build_state(mode: CpuStepMode) -> SimState {
        let mut s = SimState::new();
        s.particles.clear(); // SimState::new spawns 512 thread_rng particles
        s.params.cpu_step_mode = mode;
        s.params.reactions_enabled = false; // reactions use thread_rng
        s.params.wrap = true;
        s.params.bounds = 10.0;
        // Large enough r_max for a meaningful grid (grid_res >= 3),
        // large dt/force so particles leapfrog (the regime under study).
        s.params.r_max = 2.0; // scaled: 2.0 * 10/20 = 1.0 -> grid_res = 10
        s.params.dt = 0.05;
        s.params.force_scale = 8.0;

        let n = s.params.type_count;
        for i in 0..(n * n) {
            s.force_matrix[i] = hash01(i as u32) * 2.0 - 1.0;
        }
        for i in 0..3000u32 {
            let pos = Vec3::new(
                hash01(i * 3) * s.params.bounds,
                hash01(i * 3 + 1) * s.params.bounds,
                hash01(i * 3 + 2) * s.params.bounds,
            );
            s.particles.push(Particle::new(pos, i % n as u32));
        }
        s
    }

    /// GridExact must produce BIT-IDENTICAL results to Naive: candidates are
    /// sorted by index before accumulation, so the f32 sums match exactly.
    /// The period-2 oscillating objects this project studies sit on a
    /// floating-point knife edge, so "same model, different rounding" would
    /// still perturb them — this test enforces the stronger guarantee.
    #[test]
    fn grid_exact_is_bit_identical_to_naive() {
        let mut naive = build_state(CpuStepMode::Naive);
        let mut grid = build_state(CpuStepMode::GridExact);

        for step in 0..50 {
            cpu_step(&mut naive);
            cpu_step(&mut grid);
            assert!(naive.last_step_used_grid == false);
            assert!(grid.last_step_used_grid == true);

            for (i, (a, b)) in naive.particles.iter().zip(grid.particles.iter()).enumerate() {
                assert_eq!(
                    a.position.map(f32::to_bits),
                    b.position.map(f32::to_bits),
                    "position diverged at step {step}, particle {i}"
                );
                assert_eq!(
                    a.velocity.map(f32::to_bits),
                    b.velocity.map(f32::to_bits),
                    "velocity diverged at step {step}, particle {i}"
                );
            }
        }
    }


    /// Tiny wrapped grids alias neighbor cells; the dedup must prevent
    /// double-counted forces there too.
    #[test]
    fn tiny_wrapped_grid_matches_naive() {
        let mut naive = build_state(CpuStepMode::Naive);
        let mut grid = build_state(CpuStepMode::GridExact);
        for s in [&mut naive, &mut grid] {
            s.params.r_max = 8.0; // scaled 4.0 over bounds 10 -> grid_res = 2 (aliasing regime)
            s.particles.truncate(600);
        }

        for _ in 0..30 {
            cpu_step(&mut naive);
            cpu_step(&mut grid);
        }
        for (a, b) in naive.particles.iter().zip(grid.particles.iter()) {
            assert_eq!(a.position.map(f32::to_bits), b.position.map(f32::to_bits));
        }
    }
}
