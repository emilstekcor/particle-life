const MAX_DIM: u32 = 8u;

struct Particle {
    position: array<f32, MAX_DIM>,
    kind: u32,
    velocity: array<f32, MAX_DIM>,
    prefab_id: i32,
    _pad: vec2<u32>,
};

struct SelectionParams {
    view_proj: mat4x4<f32>,
    mode_flags: vec4<u32>,
    rect_min: vec4<f32>,
    rect_max: vec4<f32>,
    brush_data: vec4<f32>,
    viewport: vec4<f32>,
    slice_centers: array<vec4<f32>, 2>,
    slice_thickness: array<vec4<f32>, 2>,
    dimension_data: vec4<u32>,
    projection: array<vec4<f32>,16>,
    projection_data: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> selection: array<u32>;
@group(0) @binding(2) var<uniform> sel: SelectionParams;

fn position_component(position: array<f32, MAX_DIM>, dimension: u32) -> f32 {
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
        case 3u: { return sel.slice_centers[0].w; }
        case 4u: { return sel.slice_centers[1].x; }
        case 5u: { return sel.slice_centers[1].y; }
        case 6u: { return sel.slice_centers[1].z; }
        default: { return sel.slice_centers[1].w; }
    }
}

fn slice_width(dimension: u32) -> f32 {
    switch dimension {
        case 3u: { return sel.slice_thickness[0].w; }
        case 4u: { return sel.slice_thickness[1].x; }
        case 5u: { return sel.slice_thickness[1].y; }
        case 6u: { return sel.slice_thickness[1].z; }
        default: { return sel.slice_thickness[1].w; }
    }
}

fn inside_extra_slice(position: array<f32, MAX_DIM>) -> bool {
    let dimension = min(sel.dimension_data.x, MAX_DIM);
    for (var d = 3u; d < dimension; d++) {
        let center = slice_center(d);
        let half = max(slice_width(d), 0.0) * 0.5;
        if abs(position_component(position, d) - center) >= half {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sel.mode_flags.y { return; }

    let mode = sel.mode_flags.x;
    if mode == 0u { return; }

    var full_position = project_nd(particles[i].position);
    let position = vec3<f32>(full_position[0], full_position[1], full_position[2]);
    var hit = false;

    if mode == 1u || mode == 2u {
        let clip = sel.view_proj * vec4<f32>(position, 1.0);
        if clip.w>0.0 && clip.z>=0.0 && clip.z<=clip.w {
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
        hit = abs(position.z - sel.brush_data.w) <= sel.brush_data.z * 0.5;
    }

    hit = hit && inside_extra_slice(full_position);
    if mode == 2u {
        if hit { selection[i] = 1u; }
    } else {
        selection[i] = u32(hit);
    }
}

fn project_nd(input:array<f32,8>)->array<f32,8> {
    var source=input;var result:array<f32,8>;var basis=sel.projection;
    let center=sel.projection_data.x;
    for(var row=0u;row<8u;row++) {
        var value=0.0;
        for(var col=0u;col<sel.dimension_data.x;col++){value+=basis[row*2u+col/4u][col%4u]*(source[col]-center);}
        result[row]=value+center;
    }
    return result;
}
