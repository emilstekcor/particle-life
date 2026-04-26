struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
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

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) age_alpha: f32,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<storage, read> trail_history: array<TrailPoint>;

@group(0) @binding(2)
var<uniform> trail: TrailParams;

fn slot_for_logical_index(k: u32) -> u32 {
    return (trail.head + 1u + k) % trail.trail_len;
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;

    if (trail.enabled == 0u || trail.valid_len == 0u) {
        out.pos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.age_alpha = 0.0;
        return out;
    }

    let particle_id = vid / trail.valid_len;
    let sample_id = vid % trail.valid_len;

    if (particle_id >= trail.particle_count) {
        out.pos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.age_alpha = 0.0;
        return out;
    }

    let slot = slot_for_logical_index(sample_id);
    let idx = particle_id * trail.trail_len + slot;

    let trail_point = trail_history[idx];
    
    // Apply type filter
    if (trail.type_filter >= 0 && i32(trail_point.kind) != trail.type_filter) {
        out.pos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.age_alpha = 0.0;
        return out;
    }

    let world_pos = trail_point.position;
    out.pos = camera.view_proj * vec4<f32>(world_pos, 1.0);

    let denom = max(1.0, f32(trail.valid_len));
    out.age_alpha = (f32(sample_id) + 1.0) / denom;

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let a = 0.08 + 0.45 * in.age_alpha;
    return vec4<f32>(0.9, 0.9, 1.0, a);
}
