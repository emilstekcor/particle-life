// Dedicated selection pass.
//
// Runs every frame (even while paused, and regardless of CPU/GPU physics)
// against the current particle buffer. Rect and brush selection project each
// particle into screen space (egui points) using the camera view-proj matrix,
// so what you drag on screen is what gets selected.
//
// Semantics:
//   mode 0 (none)  -> no-op: existing selection flags persist
//   mode 1 (rect)  -> replace: flag = inside rect this frame
//   mode 2 (brush) -> accumulate: painting only ever adds (stroke is cleared
//                     CPU-side when a new stroke starts)
//   mode 3 (slice) -> replace: flag = |z - center| <= thickness / 2

struct Particle {
    position: vec3<f32>,
    kind:     u32,
    velocity: vec3<f32>,
    prefab_id: i32,
};

struct SelectionParams {
    view_proj:  mat4x4<f32>,
    mode_flags: vec4<u32>,  // x = mode, y = particle count
    rect_min:   vec4<f32>,  // xy = rect min (points)
    rect_max:   vec4<f32>,  // xy = rect max (points)
    brush_data: vec4<f32>,  // rect/brush: xy = center (points), z = radius (points)
                            // slice:      z = thickness (world), w = center (world z)
    viewport:   vec4<f32>,  // xy = viewport size in egui points
};

@group(0) @binding(0) var<storage, read>       particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> selection: array<u32>;
@group(0) @binding(2) var<uniform>             sel:       SelectionParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sel.mode_flags.y {
        return;
    }

    let mode = sel.mode_flags.x;
    if mode == 0u {
        return; // keep whatever is already selected
    }

    let pos = particles[i].position;
    var hit = false;

    if mode == 1u || mode == 2u {
        // Project world position -> egui screen points
        let clip = sel.view_proj * vec4<f32>(pos, 1.0);
        if clip.w > 0.0 {
            let ndc = clip.xyz / clip.w;
            let sx = (ndc.x * 0.5 + 0.5) * sel.viewport.x;
            let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * sel.viewport.y;

            if mode == 1u {
                hit = sx >= sel.rect_min.x && sx <= sel.rect_max.x &&
                      sy >= sel.rect_min.y && sy <= sel.rect_max.y;
            } else {
                hit = distance(vec2<f32>(sx, sy), sel.brush_data.xy) <= sel.brush_data.z;
            }
        }
    } else if mode == 3u {
        hit = abs(pos.z - sel.brush_data.w) <= sel.brush_data.z * 0.5;
    }

    if mode == 2u {
        // Brush accumulates along the stroke
        if hit {
            selection[i] = 1u;
        }
    } else {
        selection[i] = u32(hit);
    }
}
