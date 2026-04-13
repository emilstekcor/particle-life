// particle_physics.wgsl
// N² particle life physics — one thread per particle.
// This is the GPU equivalent of sim/physics.rs cpu_step().

// ── Uniforms (params uploaded once per step if changed) ───────────────────────
struct Params {
    particle_count: u32,
    type_count: u32,
    r_max: f32,
    force_scale: f32,
    friction: f32,
    dt: f32,
    bounds: f32,
    wrap: u32,  // 1 = wrap, 0 = no wrap (bools aren't great in WGSL uniforms)
}

// ── Particle layout (must match Rust Particle repr(C)) ────────────────────────
struct Particle {
    position: vec3<f32>,
    kind: u32,
    velocity: vec3<f32>,
    prefab_id: i32,
}

// ── Bindings ──────────────────────────────────────────────────────────────────
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       particles_in:   array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_out:  array<Particle>;
@group(0) @binding(3) var<storage, read>       force_matrix:   array<f32>;

// ── Force function ────────────────────────────────────────────────────────────
// Matches the C++ and CPU Rust implementations exactly.
// dn = normalized distance [0, 1] where 1 = r_max
// beta = inner repulsion zone boundary
fn force_func(dn: f32, attr: f32) -> f32 {
    let beta = 0.3;
    if dn < beta {
        return dn / beta - 1.0;
    }
    return attr * (1.0 - abs(2.0 * dn - 1.0 - beta) / (1.0 - beta));
}

// ── Wrap distance helper ──────────────────────────────────────────────────────
fn wrap_delta(d: f32, bounds: f32) -> f32 {
    let half = bounds * 0.5;
    if d > half { return d - bounds; }
    if d < -half { return d + bounds; }
    return d;
}

// ── Main compute kernel ───────────────────────────────────────────────────────
// Dispatch with (particle_count / 64 + 1, 1, 1) workgroups
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= params.particle_count { return; }

    var pi = particles_in[i];

    var fx = 0.0;
    var fy = 0.0;
    var fz = 0.0;

    let r_max_sq = params.r_max * params.r_max;

    // Loop over all other particles — pure N²
    for (var j = 0u; j < params.particle_count; j++) {
        if j == i { continue; }

        let pj = particles_in[j];

        var dx = pj.position.x - pi.position.x;
        var dy = pj.position.y - pi.position.y;
        var dz = pj.position.z - pi.position.z;

        if params.wrap == 1u {
            dx = wrap_delta(dx, params.bounds);
            dy = wrap_delta(dy, params.bounds);
            dz = wrap_delta(dz, params.bounds);
        }

        let dist_sq = dx * dx + dy * dy + dz * dz;
        if dist_sq < 1e-10 || dist_sq > r_max_sq { continue; }

        let dist = sqrt(dist_sq);
        let dn = dist / params.r_max;

        // Look up force matrix for this type pair
        let attr = force_matrix[pi.kind * params.type_count + pj.kind];
        let a = force_func(dn, attr);

        let inv_dist = 1.0 / dist;
        fx += a * dx * inv_dist;
        fy += a * dy * inv_dist;
        fz += a * dz * inv_dist;
    }

    // Update velocity with damping
    // damping = friction^(dt * 60) — frame-rate independent
    let damping = pow(params.friction, params.dt * 60.0);

    pi.velocity.x = (pi.velocity.x + fx * params.force_scale * params.dt) * damping;
    pi.velocity.y = (pi.velocity.y + fy * params.force_scale * params.dt) * damping;
    pi.velocity.z = (pi.velocity.z + fz * params.force_scale * params.dt) * damping;

    // Update position
    pi.position.x += pi.velocity.x * params.dt;
    pi.position.y += pi.velocity.y * params.dt;
    pi.position.z += pi.velocity.z * params.dt;

    // Wrap position
    if params.wrap == 1u {
        if pi.position.x < 0.0 { pi.position.x += params.bounds; }
        else if pi.position.x >= params.bounds { pi.position.x -= params.bounds; }

        if pi.position.y < 0.0 { pi.position.y += params.bounds; }
        else if pi.position.y >= params.bounds { pi.position.y -= params.bounds; }

        if pi.position.z < 0.0 { pi.position.z += params.bounds; }
        else if pi.position.z >= params.bounds { pi.position.z -= params.bounds; }
    }

    particles_out[i] = pi;
}
