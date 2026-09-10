const MAX_DIM: u32 = 8u;

struct GpuParticle {
    position: array<f32, MAX_DIM>,
    kind: u32,
    velocity: array<f32, MAX_DIM>,
    prefab_id: i32,
    _pad: vec2<u32>,
};

struct TrailPoint {
    position: array<f32, MAX_DIM>,
    kind: u32,
    timer: u32,
    _pad: vec2<u32>,
};

struct TrailParams {
    particle_count: u32,
    trail_len: u32,
    head: u32,
    valid_len: u32,
    enabled: u32,
    type_filter: i32,
    trigger_only: u32,
    fade_alpha: f32,
};

@group(0) @binding(0) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(1) var<storage, read_write> trail_history: array<TrailPoint>;
@group(0) @binding(2) var<uniform> trail: TrailParams;
@group(0) @binding(3) var<storage, read> trace_timers: array<u32>;

fn write_empty(destination: u32) {
    var empty: array<f32, MAX_DIM>;
    for (var d = 0u; d < MAX_DIM; d++) {
        empty[d] = 999999.0;
    }
    trail_history[destination].position = empty;
    trail_history[destination].kind = 0u;
    trail_history[destination].timer = 0u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if trail.enabled == 0u || i >= trail.particle_count { return; }
    let destination = i * trail.trail_len + trail.head;

    if trail.type_filter >= 0 && i32(particles[i].kind) != trail.type_filter {
        write_empty(destination);
        return;
    }
    if trail.trigger_only != 0u && trace_timers[i] == 0u {
        write_empty(destination);
        return;
    }

    trail_history[destination].position = particles[i].position;
    trail_history[destination].kind = particles[i].kind;
    trail_history[destination].timer = trace_timers[i];
}
