struct GpuParticle {
    position: vec3<f32>,
    kind: u32,
    velocity: vec3<f32>,
    prefab_id: i32,
};

struct TrailPoint {
    pos_radius: vec4<f32>,    // xyz + radius/type/etc
    color_timer: vec4<u32>,  // color + timer data
};

struct TrailParams {
    particle_count: u32,
    trail_len: u32,
    head: u32,
    valid_len: u32,
    enabled: u32,
    type_filter: i32,
    trigger_only: u32,
    _pad0: u32,
};

@group(0) @binding(0)
var<storage, read> particles: array<GpuParticle>;

@group(0) @binding(1)
var<storage, read_write> trail_history: array<TrailPoint>;

@group(0) @binding(2)
var<uniform> trail: TrailParams;

@group(0) @binding(3)
var<storage, read> trace_timers: array<u32>;

fn write_empty(dst: u32) {
    trail_history[dst].pos_radius = vec4<f32>(999999.0, 999999.0, 999999.0, 0.0);
    trail_history[dst].color_timer = vec4<u32>(0u, 0u, 0u, 0u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if trail.enabled == 0u || i >= trail.particle_count {
        return;
    }

    let dst = i * trail.trail_len + trail.head;

    if trail.type_filter >= 0 && i32(particles[i].kind) != trail.type_filter {
        write_empty(dst);
        return;
    }

    if trail.trigger_only != 0u && trace_timers[i] == 0u {
        write_empty(dst);
        return;
    }

    trail_history[dst].pos_radius = vec4<f32>(particles[i].position, f32(particles[i].kind));
    trail_history[dst].color_timer = vec4<u32>(particles[i].kind, 0u, 0u, 0u);
}
