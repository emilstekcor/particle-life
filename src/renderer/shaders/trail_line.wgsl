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
    particle_size: f32,
    render_data: vec4<f32>,
    slice_centers: array<vec4<f32>, 2>,
    slice_thickness: array<vec4<f32>, 2>,
    dimension_data: vec4<u32>,
    projection: array<vec4<f32>,16>,
    projection_data: vec4<f32>,
};

struct TrailPoint {
    position: array<f32, 8>,
    kind: u32,
    timer: u32,
    _pad: vec2<u32>,
};

// Must match Rust TrailParams exactly (32 bytes).
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

@group(0) @binding(0) var<uniform>       camera:        CameraUniform;
@group(0) @binding(1) var<storage, read> trail_history: array<TrailPoint>;
@group(0) @binding(2) var<uniform>       trail:         TrailParams;

// Maps logical index k (0 = oldest, valid_len-1 = newest) to ring-buffer slot.
// After advance_trail_head(), trail.head is the slot just written (newest).
// While the buffer is still filling (valid_len < trail_len), the oldest live
// sample is NOT at head + 1 — that slot has never been written this cycle and
// still holds stale/zeroed data. The oldest live sample is `valid_len - 1`
// slots behind head, so it sits at (head + 1 + unfilled) % trail_len, where
// `unfilled` is the number of never-yet-written slots ahead of it.
fn slot_for_logical(k: u32) -> u32 {
    let unfilled = trail.trail_len - trail.valid_len;
    return (trail.head + 1u + unfilled + k) % trail.trail_len;
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

fn position_component(position: array<f32, 8>, dimension: u32) -> f32 {
    switch dimension {
        case 3u: { return position[3]; }
        case 4u: { return position[4]; }
        case 5u: { return position[5]; }
        case 6u: { return position[6]; }
        default: { return position[7]; }
    }
}

fn slice_center(dimension: u32) -> f32 {
    switch dimension {
        case 3u: { return camera.slice_centers[0].w; }
        case 4u: { return camera.slice_centers[1].x; }
        case 5u: { return camera.slice_centers[1].y; }
        case 6u: { return camera.slice_centers[1].z; }
        default: { return camera.slice_centers[1].w; }
    }
}

fn slice_width(dimension: u32) -> f32 {
    switch dimension {
        case 3u: { return camera.slice_thickness[0].w; }
        case 4u: { return camera.slice_thickness[1].x; }
        case 5u: { return camera.slice_thickness[1].y; }
        case 6u: { return camera.slice_thickness[1].z; }
        default: { return camera.slice_thickness[1].w; }
    }
}

fn extra_slice_alpha(position: array<f32, 8>) -> f32 {
    var alpha = 1.0;
    let dimension = min(camera.dimension_data.x, 8u);
    for (var d = 3u; d < dimension; d++) {
        let center = slice_center(d);
        let half = max(slice_width(d) * 0.5, 1e-6);
        let distance = abs(position_component(position, d) - center);
        alpha = min(alpha, 1.0 - smoothstep(half * 0.8, half, distance));
    }
    return alpha;
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
    let other_slot=slot_for_logical(segment_id+(1u-endpoint));
    let other=trail_history[particle_id*trail.trail_len+other_slot];
    if other.position[0]>900000.0 {return make_degenerate();}
    if camera.projection_data.y>0.0 {var a=point.position;var b=other.position;for(var d=0u;d<camera.dimension_data.x;d++){if abs(a[d]-b[d])>camera.projection_data.x{return make_degenerate();}}}


    // Inactive/filtered particles write sentinel position 999999
    if (point.position[0] > 900000.0) {
        return make_degenerate();
    }
    if (trail.type_filter >= 0 && i32(point.kind) != trail.type_filter) {
        return make_degenerate();
    }

    var out: VsOut;
    var projected=project_nd(point.position);
    out.pos=camera.view_proj*vec4<f32>(projected[0],projected[1],projected[2],1.0);

    // age_t: 0 = oldest segment, 1 = newest segment
    let age_t = f32(segment_id + 1u) / f32(valid_segments);
    let base  = type_color(point.kind);
    let alpha = (0.05 + 0.60 * age_t) * trail.fade_alpha * extra_slice_alpha(projected);
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

    if (point.position[0] > 900000.0) {
        return make_degenerate();
    }
    if (trail.type_filter >= 0 && i32(point.kind) != trail.type_filter) {
        return make_degenerate();
    }

    var out: VsOut;
    var projected=project_nd(point.position);
    out.pos=camera.view_proj*vec4<f32>(projected[0],projected[1],projected[2],1.0);

    // age_t: 0 = oldest, 1 = newest
    let age_t = (f32(sample_id) + 1.0) / f32(trail.valid_len);
    let base  = type_color(point.kind);
    let alpha = (0.08 + 0.55 * age_t) * trail.fade_alpha * extra_slice_alpha(projected);
    out.color = vec4<f32>(base * age_t, alpha);

    return out;
}

@fragment
fn fs_point(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

fn project_nd(input:array<f32,8>)->array<f32,8> {
    var source=input;var result:array<f32,8>;var basis=camera.projection;
    let center=camera.projection_data.x;
    for(var row=0u;row<8u;row++) {
        var value=0.0;
        for(var col=0u;col<camera.dimension_data.x;col++){value+=basis[row*2u+col/4u][col%4u]*(source[col]-center);}
        result[row]=value+center;
    }
    return result;
}
