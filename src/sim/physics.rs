use crate::sim::SimState;

/// CPU N² physics step — this is the reference implementation.
/// The compute shader in renderer/compute.rs does the same thing on the GPU.
/// We keep this so you can verify GPU output matches, and as a fallback.
pub fn cpu_step(state: &mut SimState) {
    let count = state.particles.len();
    if count == 0 {
        return;
    }

    let dt = state.params.dt;
    let r_max = state.params.r_max;
    let r_max_sq = r_max * r_max;
    let force_scale = state.params.force_scale;
    let friction = state.params.friction;
    let wrap = state.params.wrap;
    let bounds = state.params.bounds;
    let beta = 0.3_f32; // inner repulsion zone (matches C++ version)
    let type_count = state.params.type_count;

    // Damping factor matches C++ version: pow(friction, dt * 60)
    let damping = friction.powf(dt * 60.0);

    // Accumulate forces into a separate velocity buffer
    let mut new_velocities: Vec<[f32; 3]> = state.particles
        .iter()
        .map(|p| p.velocity)
        .collect();

    for i in 0..count {
        let pi_pos = state.particles[i].position;
        let pi_kind = state.particles[i].kind as usize;

        let mut fx = 0.0_f32;
        let mut fy = 0.0_f32;
        let mut fz = 0.0_f32;

        for j in 0..count {
            if i == j {
                continue;
            }

            let pj_pos = state.particles[j].position;
            let pj_kind = state.particles[j].kind as usize;

            let mut dx = pj_pos[0] - pi_pos[0];
            let mut dy = pj_pos[1] - pi_pos[1];
            let mut dz = pj_pos[2] - pi_pos[2];

            // Wrap distance (toroidal)
            if wrap {
                let half = bounds * 0.5;
                if dx > half { dx -= bounds; } else if dx < -half { dx += bounds; }
                if dy > half { dy -= bounds; } else if dy < -half { dy += bounds; }
                if dz > half { dz -= bounds; } else if dz < -half { dz += bounds; }
            }

            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < 1e-10 || dist_sq > r_max_sq {
                continue;
            }

            let dist = dist_sq.sqrt();
            let dn = dist / r_max; // normalized distance [0, 1]

            // Force function (matches C++ particle_step_kernel exactly)
            // Inner zone (dn < beta): universal repulsion
            // Outer zone: type-specific attraction/repulsion
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

        let vel = &mut new_velocities[i];
        vel[0] = (vel[0] + fx * force_scale * dt) * damping;
        vel[1] = (vel[1] + fy * force_scale * dt) * damping;
        vel[2] = (vel[2] + fz * force_scale * dt) * damping;
    }

    // Apply velocities and wrap positions
    for (i, p) in state.particles.iter_mut().enumerate() {
        p.velocity = new_velocities[i];
        p.position[0] += p.velocity[0] * dt;
        p.position[1] += p.velocity[1] * dt;
        p.position[2] += p.velocity[2] * dt;

        if wrap {
            for coord in &mut p.position {
                if *coord < 0.0 { *coord += bounds; }
                else if *coord >= bounds { *coord -= bounds; }
            }
        }
    }
}
