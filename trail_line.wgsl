// trail_line.wgsl
// Handles both trail rendering modes:
//   vs_main / fs_main  — LineList segments (Trail Lines mode)
//   vs_point / fs_point — PointList dots  (Trail Dots mode)
//
// Bind group layout (both pipelines share it):
//   binding 0 = CameraUniform  (uniform)
//   binding 1 = trail_history  (storage, read)
//   binding 2 = TrailParams    (uniform)

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    _padding: f32,
};

struct TrailPoint {
    position: vec3<f32>,
    kind: u32,
    _pad: u32,  // Rust TrailPoint = { [f32;3], u32, u32 } = 20 bytes
};

// Must match Rust TrailParams exactly (8 × u32 = 32 bytes)
struct TrailParams {
    particle_count: u32,
    trail_len: u32,
    head: u32,
    valid_len: u32,
    enabled: u32,
    type_filter: i32,
    trigger_only: u32,  // previously mis-labelled _pad0 in the old shader
    _pad0: u32,
};

@group(0) @binding(0) var<uniform>       camera:        CameraUniform;
@group(0) @binding(1) var<storage, read> trail_history: array<TrailPoint>;
@group(0) @binding(2) var<uniform>       trail:         TrailParams;

// Maps logical index k (0 = oldest, valid_len-1 = newest) to ring-buffer slot.
// After advance_trail_head(), trail.head is the slot just written (newest),
// so oldest is at (head + 1) % trail_len.
fn slot_for_logical(k: u32) -> u32 {
    return (trail.head + 1u + k) % trail.trail_len;
}

fn type_color(kind: u32) -> vec3<f32> {
    let i = kind % 8u;
    if (i == 0u) { return vec3(0.96, 0.36, 0.36); }
    if (i == 1u) { return vec3(0.36, 0.76, 0.96); }
    if (i == 2u) { return vec3(0.56, 0.96, 0.36); }
    if (i == 3u) { return vec3(0.96, 0.76, 0.26); }
    if (i == 4u) { return vec3(0.86, 0.46, 0.96); }
    if (i == 5u) { return vec3(0.96, 0.56, 0.26); }
    if (i == 6u) { return vec3(0.36, 0.96, 0.76); }
    return vec3(0.96, 0.76, 0.86);
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

fn make_degenerate() -> VsOut {
    var out: VsOut;
    out.pos   = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    out.color = vec4<f32>(0.0);
    return out;
}

// ─── LINE LIST ────────────────────────────────────────────────────────────────
//
// CPU issues: particle_count * (valid_len - 1) * 2  vertices.
//
// Layout for one particle (valid_segments = valid_len - 1):
//   seg 0 → vid 0 (older end), vid 1 (newer end)
//   seg 1 → vid 2 (older end), vid 3 (newer end)
//   ...
// Particles are concatenated sequentially.

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    if (trail.enabled == 0u || trail.valid_len < 2u) {
        return make_degenerate();
    }

    let valid_segments     = trail.valid_len - 1u;
    let verts_per_particle = valid_segments * 2u;

    let particle_id  = vid / verts_per_particle;
    let local_vid    = vid % verts_per_particle;
    let segment_id   = local_vid / 2u;  // which segment: 0..valid_segments-1
    let endpoint     = local_vid % 2u;  // 0 = older end, 1 = newer end

    if (particle_id >= trail.particle_count) {
        return make_degenerate();
    }

    // Older endpoint = logical index segment_id
    // Newer endpoint = logical index segment_id + 1
    let logical_index = segment_id + endpoint;
    let slot = slot_for_logical(logical_index);
    let idx  = particle_id * trail.trail_len + slot;

    let point = trail_history[idx];

    // Inactive/filtered particles write sentinel position 999999
    if (point.position.x > 900000.0) {
        return make_degenerate();
    }
    if (trail.type_filter >= 0 && i32(point.kind) != trail.type_filter) {
        return make_degenerate();
    }

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(point.position, 1.0);

    // age_t: 0 = oldest segment, 1 = newest segment
    let age_t = f32(segment_id + 1u) / f32(valid_segments);
    let base  = type_color(point.kind);
    let alpha = 0.05 + 0.60 * age_t;
    out.color = vec4<f32>(base * (0.3 + 0.7 * age_t), alpha);

    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

// ─── POINT LIST ───────────────────────────────────────────────────────────────
//
// CPU issues: particle_count * valid_len  vertices.
//
// Layout:
//   particle 0: vid 0 .. valid_len-1
//   particle 1: vid valid_len .. 2*valid_len-1
//   ...

@vertex
fn vs_point(@builtin(vertex_index) vid: u32) -> VsOut {
    if (trail.enabled == 0u || trail.valid_len == 0u) {
        return make_degenerate();
    }

    let particle_id = vid / trail.valid_len;
    let sample_id   = vid % trail.valid_len;  // 0 = oldest, valid_len-1 = newest

    if (particle_id >= trail.particle_count) {
        return make_degenerate();
    }

    let slot = slot_for_logical(sample_id);
    let idx  = particle_id * trail.trail_len + slot;

    let point = trail_history[idx];

    if (point.position.x > 900000.0) {
        return make_degenerate();
    }
    if (trail.type_filter >= 0 && i32(point.kind) != trail.type_filter) {
        return make_degenerate();
    }

    var out: VsOut;
    out.pos = camera.view_proj * vec4<f32>(point.position, 1.0);

    // age_t: 0 = oldest, 1 = newest
    let age_t = (f32(sample_id) + 1.0) / f32(trail.valid_len);
    let base  = type_color(point.kind);
    let alpha = 0.08 + 0.55 * age_t;
    out.color = vec4<f32>(base * age_t, alpha);

    return out;
}

@fragment
fn fs_point(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
