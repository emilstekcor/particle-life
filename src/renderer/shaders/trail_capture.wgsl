struct GpuParticle {
    position: vec3<f32>,
    kind: u32,
    velocity: vec3<f32>,
    prefab_id: i32,
};

struct TrailPoint {
    position: vec3<f32>,
    kind: u32,
    _pad: u32,  // Ensure 16-byte alignment for GPU buffers
};

struct TrailParams {
    particle_count: u32,
    trail_len: u32,
    head: u32,
    valid_len: u32,
    enabled: u32,
    type_filter: i32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<storage, read> particles: array<GpuParticle>;

@group(0) @binding(1)
var<storage, read_write> trail_history: array<TrailPoint>;

@group(0) @binding(2)
var<uniform> trail: TrailParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (trail.enabled == 0u || i >= trail.particle_count) {
        return;
    }

    let dst = i * trail.trail_len + trail.head;
    trail_history[dst].position = particles[i].position;
    trail_history[dst].kind = particles[i].kind;
}
