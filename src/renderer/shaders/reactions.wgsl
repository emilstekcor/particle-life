struct Particle {
    position: vec3<f32>,
    kind:     u32,
    velocity: vec3<f32>,
    prefab_id: i32,
};

struct ReactionParams {
    mix_radius:   f32,
    mix_radius_sq: f32,
    prob_frame:   f32,  // pre-computed: 1 - (1-p)^(dt*60)
    pass_index:   u32,  // 0..7
    type_count:   u32,
    particle_count: u32,
    bounds:       f32,
    wrap:         u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform>             params:    ReactionParams;
@group(0) @binding(2) var<storage, read>       reaction_table: array<i32>; // [type_count * type_count]
@group(0) @binding(3) var<storage, read_write> rng_state: array<u32>;      // one seed per particle

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particle_count { return; }

    let p = particles[i];

    // compute cell coords and parity — skip if not this pass
    let grid_res = u32(ceil(params.bounds / params.mix_radius));
    let cell = vec3<u32>(
        u32(p.position.x / params.bounds * f32(grid_res)),
        u32(p.position.y / params.bounds * f32(grid_res)),
        u32(p.position.z / params.bounds * f32(grid_res)),
    );
    let parity = (cell.x + cell.y + cell.z) % 8u;
    if parity != params.pass_index { return; }

    // scan all particles for a reaction candidate
    // (swap for grid lookup once working)
    for (var j = 0u; j < params.particle_count; j++) {
        if j == i { continue; }

        var d = particles[j].position - p.position;
        if params.wrap != 0u {
            let b = params.bounds;
            if d.x >  b*0.5 { d.x -= b; }
            if d.x < -b*0.5 { d.x += b; }
            if d.y >  b*0.5 { d.y -= b; }
            if d.y < -b*0.5 { d.y += b; }
            if d.z >  b*0.5 { d.z -= b; }
            if d.z < -b*0.5 { d.z += b; }
        }
        if dot(d, d) > params.mix_radius_sq { continue; }

        let result = reaction_table[p.kind * params.type_count + particles[j].kind];
        if result < 0 { continue; }

        // pcg random
        var seed = rng_state[i];
        seed = seed * 747796405u + 2891336453u;
        let r = f32((seed >> 17u) & 0xffffu) / 65535.0;
        rng_state[i] = seed;
        if r > params.prob_frame { continue; }

        particles[i].kind = u32(result);
        particles[j].kind = u32(result);  // preserve_particle_count=true for now
        return; // one reaction per particle per frame
    }
}
