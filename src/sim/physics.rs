use crate::sim::{CpuStepMode, SimState};
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
/// GridExact keeps the same force law and integration behavior.
/// It is not bit-identical to naive because accumulation order differs,
/// but it is the same interaction model.
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
            if count >= state.params.auto_grid_threshold && state.params.r_max > 0.0 && state.params.bounds > 0.0 {
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
    let r_max = state.params.r_max;
    let r_max_sq = r_max * r_max;
    let force_scale = state.params.force_scale;
    let friction = state.params.friction;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;
    let beta = state.params.beta;
    let max_speed = state.params.max_speed;
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
                let attr = state.force_matrix[pi_kind * type_count + pj_kind];
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
    let r_max = state.params.r_max;
    let r_max_sq = r_max * r_max;
    let force_scale = state.params.force_scale;
    let friction = state.params.friction;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;
    let beta = state.params.beta;
    let max_speed = state.params.max_speed;
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

    for i in 0..count {
        let pi_pos = state.particles[i].position;
        let pi_kind = state.particles[i].kind as usize;
        let [cx, cy, cz] = cell_coords(pi_pos, bounds, grid_res, wrap);

        let mut fx = 0.0_f32;
        let mut fy = 0.0_f32;
        let mut fz = 0.0_f32;

        for ox in -1isize..=1 {
            let Some(nx) = neighbor_axis(cx, ox, grid_res, wrap) else { continue; };
            for oy in -1isize..=1 {
                let Some(ny) = neighbor_axis(cy, oy, grid_res, wrap) else { continue; };
                for oz in -1isize..=1 {
                    let Some(nz) = neighbor_axis(cz, oz, grid_res, wrap) else { continue; };

                    let nid = cell_index(nx, ny, nz, grid_res);
                    for &j in &state.buckets_scratch[nid] {
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
                            let attr = state.force_matrix[pi_kind * type_count + pj_kind];
                            attr * (1.0 - ((2.0 * dn - 1.0 - beta) / (1.0 - beta)).abs())
                        };

                        let inv_dist = 1.0 / dist;
                        fx += a * dx * inv_dist;
                        fy += a * dy * inv_dist;
                        fz += a * dz * inv_dist;
                    }
                }
            }
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
