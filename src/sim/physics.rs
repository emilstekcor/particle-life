use crate::sim::{CellCoord, CpuStepMode, SimState, VecNd, MAX_DIM};
use std::time::Instant;

/// Above five dimensions, 3^D neighboring cells cost more than the sparse
/// grid normally saves. Physics remains exact by falling back to the naive
/// pair scan.
const MAX_GRID_DIM: usize = 5;

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

pub fn cpu_step(state: &mut SimState) {
    let frame_start = Instant::now();
    let count = state.particles.len();
    state.trace_timers.resize(count, 0);
    for t in &mut state.trace_timers {
        *t = t.saturating_sub(1);
    }
    if count == 0 {
        state.last_step_used_grid = false;
        state.last_neighbor_checks = 0;
        state.last_grid_res = 1;
        state.last_step_ms = 0.0;
        return;
    }

    let stats = match resolve_step_mode(state, count) {
        ResolvedStepMode::Naive => cpu_step_naive(state),
        ResolvedStepMode::GridExact => cpu_step_grid_exact(state),
    };

    apply_reactions(state);
    state.last_step_used_grid = stats.used_grid;
    state.last_neighbor_checks = stats.neighbor_checks;
    state.last_grid_res = stats.grid_res;
    state.last_step_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
    state.avg_step_ms = if state.avg_step_ms == 0.0 {
        state.last_step_ms
    } else {
        state.avg_step_ms * 0.95 + state.last_step_ms * 0.05
    };
}

fn resolve_step_mode(state: &SimState, count: usize) -> ResolvedStepMode {
    if state.params.dimension > MAX_GRID_DIM {
        return ResolvedStepMode::Naive;
    }
    match state.params.cpu_step_mode {
        CpuStepMode::Naive => ResolvedStepMode::Naive,
        CpuStepMode::GridExact => ResolvedStepMode::GridExact,
        CpuStepMode::Auto => {
            if count >= state.params.auto_grid_threshold {
                ResolvedStepMode::GridExact
            } else {
                ResolvedStepMode::Naive
            }
        }
    }
}

fn cpu_step_naive(state: &mut SimState) -> StepStats {
    let count = state.particles.len();
    let mut velocities = std::mem::take(&mut state.vel_scratch);
    velocities.resize(count, [0.0; MAX_DIM]);
    let mut neighbor_checks = 0;

    for i in 0..count {
        velocities[i] = compute_velocity(state, i, 0..count, &mut neighbor_checks);
    }
    integrate(state, &velocities);
    state.vel_scratch = velocities;

    StepStats {
        used_grid: false,
        neighbor_checks,
        grid_res: 1,
    }
}

fn cpu_step_grid_exact(state: &mut SimState) -> StepStats {
    let count = state.particles.len();
    let dim = state.params.dimension;
    let bounds = state.params.bounds;
    let wrap = state.params.wrap;
    let grid_res = choose_grid_res(bounds, state.params.scaled_r_max());

    let mut buckets = std::mem::take(&mut state.buckets_scratch);
    buckets.clear();
    for (i, particle) in state.particles.iter().enumerate() {
        buckets
            .entry(cell_coords(particle.position, bounds, grid_res, wrap, dim))
            .or_default()
            .push(i);
    }

    let mut velocities = std::mem::take(&mut state.vel_scratch);
    velocities.resize(count, [0.0; MAX_DIM]);
    let mut neighbor_checks = 0;
    let mut candidates = Vec::with_capacity(256);
    let mut keys = Vec::with_capacity(3usize.pow(dim as u32));

    for i in 0..count {
        let center = cell_coords(state.particles[i].position, bounds, grid_res, wrap, dim);
        neighbor_keys(center, dim, grid_res, wrap, &mut keys);
        candidates.clear();
        for key in &keys {
            if let Some(indices) = buckets.get(key) {
                candidates.extend_from_slice(indices);
            }
        }

        // Matching the naive ascending-index accumulation order makes GridExact
        // bit-identical, including on tiny wrapped grids where cells alias.
        candidates.sort_unstable();
        candidates.dedup();
        velocities[i] =
            compute_velocity(state, i, candidates.iter().copied(), &mut neighbor_checks);
    }

    integrate(state, &velocities);
    state.vel_scratch = velocities;
    state.buckets_scratch = buckets;

    StepStats {
        used_grid: true,
        neighbor_checks,
        grid_res,
    }
}

fn compute_velocity(
    state: &SimState,
    i: usize,
    candidates: impl IntoIterator<Item = usize>,
    neighbor_checks: &mut u64,
) -> VecNd {
    let dim = state.params.dimension;
    let pi = &state.particles[i];
    let r_max = state.params.scaled_r_max();
    let r_max_sq = r_max * r_max;
    let mut force = [0.0; MAX_DIM];

    if r_max > 0.0 {
        for j in candidates {
            *neighbor_checks += 1;
            if i == j {
                continue;
            }

            let pj = &state.particles[j];
            let mut delta = [0.0; MAX_DIM];
            let mut dist_sq = 0.0;
            for d in 0..dim {
                let raw = pj.position[d] - pi.position[d];
                delta[d] = if state.params.wrap {
                    wrap_delta(raw, state.params.bounds)
                } else {
                    raw
                };
                dist_sq += delta[d] * delta[d];
            }
            if dist_sq < 1e-10 || dist_sq > r_max_sq {
                continue;
            }

            let dist = dist_sq.sqrt();
            let normalized = dist / r_max;
            let attraction = if normalized < state.params.beta {
                normalized / state.params.beta - 1.0
            } else {
                let matrix_index = pi.kind as usize * state.params.type_count + pj.kind as usize;
                let attr = state.force_matrix.get(matrix_index).copied().unwrap_or(0.0);
                attr * (1.0
                    - ((2.0 * normalized - 1.0 - state.params.beta) / (1.0 - state.params.beta))
                        .abs())
            };
            let inv_dist = 1.0 / dist;
            for d in 0..dim {
                force[d] += attraction * delta[d] * inv_dist;
            }
        }
    }

    let damping = state.params.friction.powf(state.params.dt * 60.0);
    let mut velocity = pi.velocity;
    for d in 0..dim {
        velocity[d] =
            (velocity[d] + force[d] * state.params.force_scale * state.params.dt) * damping;
    }
    clamp_velocity(&mut velocity, state.params.scaled_max_speed(), dim);
    velocity
}

fn integrate(state: &mut SimState, velocities: &[VecNd]) {
    let dim = state.params.dimension;
    let dt = state.params.dt;
    let bounds = state.params.bounds;
    let wrap = state.params.wrap;

    for (particle, velocity) in state.particles.iter_mut().zip(velocities) {
        particle.velocity = *velocity;
        for d in 0..dim {
            particle.position[d] += particle.velocity[d] * dt;
            if wrap && bounds > 0.0 {
                particle.position[d] = particle.position[d].rem_euclid(bounds);
            }
        }
    }
}

fn clamp_velocity(velocity: &mut VecNd, max_speed: f32, dim: usize) {
    let speed_sq: f32 = velocity[..dim].iter().map(|v| v * v).sum();
    if speed_sq > max_speed * max_speed && speed_sq > 0.0 {
        let scale = max_speed / speed_sq.sqrt();
        for component in velocity.iter_mut().take(dim) {
            *component *= scale;
        }
    }
}

fn choose_grid_res(bounds: f32, radius: f32) -> usize {
    if bounds <= 0.0 || radius <= 0.0 {
        return 1;
    }
    ((bounds / radius).floor() as usize).clamp(1, 200)
}

fn wrap_delta(mut value: f32, bounds: f32) -> f32 {
    let half = bounds * 0.5;
    if value > half {
        value -= bounds;
    } else if value < -half {
        value += bounds;
    }
    value
}

fn cell_coords(position: VecNd, bounds: f32, grid_res: usize, wrap: bool, dim: usize) -> CellCoord {
    let mut result = [0; MAX_DIM];
    for d in 0..dim {
        result[d] = axis_to_cell(position[d], bounds, grid_res, wrap);
    }
    result
}

fn axis_to_cell(value: f32, bounds: f32, grid_res: usize, wrap: bool) -> usize {
    if grid_res <= 1 || bounds <= 0.0 {
        return 0;
    }
    let value = if wrap {
        value.rem_euclid(bounds.max(f32::EPSILON))
    } else {
        value.clamp(0.0, (bounds - f32::EPSILON).max(0.0))
    };
    ((value / bounds) * grid_res as f32)
        .floor()
        .clamp(0.0, (grid_res - 1) as f32) as usize
}

fn neighbor_axis(axis: usize, delta: isize, grid_res: usize, wrap: bool) -> Option<usize> {
    if wrap {
        Some((axis as isize + delta).rem_euclid(grid_res as isize) as usize)
    } else {
        let value = axis as isize + delta;
        (value >= 0 && value < grid_res as isize).then_some(value as usize)
    }
}

fn neighbor_keys(
    center: CellCoord,
    dim: usize,
    grid_res: usize,
    wrap: bool,
    output: &mut Vec<CellCoord>,
) {
    output.clear();
    output.push([0; MAX_DIM]);
    for d in 0..dim {
        let previous = std::mem::take(output);
        output.reserve(previous.len() * 3);
        for base in previous {
            for offset in -1..=1 {
                if let Some(axis) = neighbor_axis(center[d], offset, grid_res, wrap) {
                    let mut key = base;
                    key[d] = axis;
                    output.push(key);
                }
            }
        }
    }
    output.sort_unstable();
    output.dedup();
}

pub fn apply_reactions(state: &mut SimState) {
    if !state.params.reactions_enabled || state.particles.is_empty() {
        return;
    }
    let radius = state.params.scaled_mix_radius();
    if radius <= 0.0 {
        return;
    }

    let count = state.particles.len();
    let dim = state.params.dimension;
    let grid_res = choose_grid_res(state.params.bounds, radius);
    let use_grid = count >= 1024 && grid_res >= 3 && dim <= MAX_GRID_DIM;
    let mut changes = std::mem::take(&mut state.reaction_changes_scratch);
    changes.clear();
    let mut traces = Vec::new();

    if use_grid {
        let mut buckets = std::mem::take(&mut state.buckets_scratch);
        buckets.clear();
        for (i, particle) in state.particles.iter().enumerate() {
            buckets
                .entry(cell_coords(
                    particle.position,
                    state.params.bounds,
                    grid_res,
                    state.params.wrap,
                    dim,
                ))
                .or_default()
                .push(i);
        }

        let mut keys = Vec::with_capacity(3usize.pow(dim as u32));
        let mut candidates = Vec::with_capacity(256);
        for i in 0..count {
            let center = cell_coords(
                state.particles[i].position,
                state.params.bounds,
                grid_res,
                state.params.wrap,
                dim,
            );
            neighbor_keys(center, dim, grid_res, state.params.wrap, &mut keys);
            candidates.clear();
            for key in &keys {
                if let Some(indices) = buckets.get(key) {
                    candidates.extend(indices.iter().copied().filter(|&j| j > i));
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            for &j in &candidates {
                maybe_react_pair(state, i, j, radius * radius, &mut changes, &mut traces);
            }
        }
        state.buckets_scratch = buckets;
    } else {
        for i in 0..count {
            for j in (i + 1)..count {
                maybe_react_pair(state, i, j, radius * radius, &mut changes, &mut traces);
            }
        }
    }

    for (i, t) in traces {
        state.trace_timers[i] = state.trace_timers[i].max(t);
    }
    for (index, kind) in changes.drain(..) {
        if let Some(particle) = state.particles.get_mut(index) {
            particle.kind = kind;
        }
    }
    state.reaction_changes_scratch = changes;
}

fn maybe_react_pair(
    state: &SimState,
    i: usize,
    j: usize,
    radius_sq: f32,
    changes: &mut Vec<(usize, u32)>,
    traces: &mut Vec<(usize, u32)>,
) {
    let a = &state.particles[i];
    let b = &state.particles[j];
    let mut distance_sq = 0.0;
    for d in 0..state.params.dimension {
        let raw = b.position[d] - a.position[d];
        let delta = if state.params.wrap {
            wrap_delta(raw, state.params.bounds)
        } else {
            raw
        };
        distance_sq += delta * delta;
    }
    if distance_sq > radius_sq {
        return;
    }

    let type_count = state.params.type_count;
    let ai = a.kind as usize;
    let bi = b.kind as usize;
    if ai >= type_count || bi >= type_count {
        return;
    }
    let result_ab = state.reaction_table[ai * type_count + bi];
    let result_ba = state.reaction_table[bi * type_count + ai];
    if result_ab >= 0
        && reaction_gate(
            i as u32,
            j as u32,
            state.step_count as u32,
            state.params.reaction_probability,
        )
    {
        changes.push((i, result_ab as u32));
        let t = state.trace_len_matrix[ai * type_count + bi];
        traces.push((i, t));
        traces.push((j, t));
    }
    if result_ba >= 0
        && reaction_gate(
            j as u32,
            i as u32,
            state.step_count as u32,
            state.params.reaction_probability,
        )
    {
        changes.push((j, result_ba as u32));
        let t = state.trace_len_matrix[bi * type_count + ai];
        traces.push((i, t));
        traces.push((j, t));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{Particle, SimState};
    use glam::Vec3;

    fn hash01(seed: u32) -> f32 {
        let mut hash = seed.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
        hash ^= hash >> 16;
        hash = hash.wrapping_mul(0x85EB_CA6B);
        hash ^= hash >> 13;
        (hash & 0x00FF_FFFF) as f32 / 16_777_216.0
    }

    fn build_state(mode: CpuStepMode, dimension: usize) -> SimState {
        let mut state = SimState::new();
        state.particles.clear();
        state.params.cpu_step_mode = mode;
        state.params.dimension = dimension;
        state.params.reactions_enabled = false;
        state.params.wrap = true;
        state.params.bounds = 10.0;
        state.params.r_max = 2.0;
        state.params.dt = 0.05;
        state.params.force_scale = 8.0;
        for i in 0..state.force_matrix.len() {
            state.force_matrix[i] = hash01(i as u32) * 2.0 - 1.0;
        }
        for i in 0..600u32 {
            let mut particle = Particle::new(
                Vec3::new(
                    hash01(i * 3) * 10.0,
                    hash01(i * 3 + 1) * 10.0,
                    hash01(i * 3 + 2) * 10.0,
                ),
                i % state.params.type_count as u32,
            );
            for d in 3..dimension {
                particle.position[d] = hash01(i * MAX_DIM as u32 + d as u32) * 10.0;
            }
            state.particles.push(particle);
        }
        state
    }

    #[test]
    fn grid_exact_matches_naive_in_3d_and_4d() {
        for dimension in [3, 4] {
            let mut naive = build_state(CpuStepMode::Naive, dimension);
            let mut grid = build_state(CpuStepMode::GridExact, dimension);
            for step in 0..5 {
                cpu_step(&mut naive);
                cpu_step(&mut grid);
                for (index, (a, b)) in naive.particles.iter().zip(&grid.particles).enumerate() {
                    assert_eq!(
                        a.position.map(f32::to_bits),
                        b.position.map(f32::to_bits),
                        "position diverged in {dimension}D at step {step}, particle {index}"
                    );
                    assert_eq!(a.velocity.map(f32::to_bits), b.velocity.map(f32::to_bits));
                }
            }
        }
    }

    #[test]
    fn constant_fourth_axis_matches_3d() {
        let mut three = build_state(CpuStepMode::Naive, 3);
        let mut four = build_state(CpuStepMode::Naive, 3);
        four.params.dimension = 4;
        for _ in 0..5 {
            cpu_step(&mut three);
            cpu_step(&mut four);
        }
        for (a, b) in three.particles.iter().zip(&four.particles) {
            assert_eq!(a.position[..3], b.position[..3]);
            assert_eq!(a.velocity[..3], b.velocity[..3]);
            assert_eq!(b.position[3], 0.0);
        }
    }

    #[test]
    fn five_dimensional_neighborhood_has_243_cells() {
        let mut keys = Vec::new();
        neighbor_keys([5; MAX_DIM], 5, 20, false, &mut keys);
        assert_eq!(keys.len(), 243);
    }

    #[test]
    fn dimensions_above_five_fall_back_to_naive() {
        let state = build_state(CpuStepMode::GridExact, 6);
        assert_eq!(
            resolve_step_mode(&state, state.particles.len()),
            ResolvedStepMode::Naive
        );
    }
}

pub fn reaction_gate(i: u32, j: u32, frame: u32, probability: f32) -> bool {
    let mut h = i
        .wrapping_mul(2654435761)
        .wrapping_add(j.wrapping_mul(2246822519))
        .wrapping_add(frame.wrapping_mul(2246822519));
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846ca68b);
    h ^= h >> 16;
    probability >= 1.0 || (h & 65535) < (probability.clamp(0.0, 1.0) * 65536.0) as u32
}
#[cfg(test)]
mod reaction_regressions {
    use super::*;
    #[test]
    fn probability_endpoints_are_exact() {
        for i in 0..1000 {
            assert!(!reaction_gate(i, i + 1, i, 0.0));
            assert!(reaction_gate(i, i + 1, i, 1.0));
        }
    }
}
