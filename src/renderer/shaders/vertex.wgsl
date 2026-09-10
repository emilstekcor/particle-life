const MAX_DIM: u32 = 8u;

struct VertexInput {
    @location(0) position_low: vec4<f32>,
    @location(1) kind: u32,
    @location(2) position_high: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) offset: vec2<f32>,
    @location(2) soft: f32,
    @location(3) slice_alpha: f32,
};

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

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<storage, read> selection: array<u32>;

fn type_color(kind: u32) -> vec3<f32> {
    let index = kind % 8u;
    if index == 0u { return vec3(0.96, 0.36, 0.36); }
    if index == 1u { return vec3(0.36, 0.76, 0.96); }
    if index == 2u { return vec3(0.56, 0.96, 0.36); }
    if index == 3u { return vec3(0.96, 0.76, 0.26); }
    if index == 4u { return vec3(0.86, 0.46, 0.96); }
    if index == 5u { return vec3(0.96, 0.56, 0.26); }
    if index == 6u { return vec3(0.36, 0.96, 0.76); }
    return vec3(0.96, 0.76, 0.86);
}

fn corner_offset(id: u32) -> vec2<f32> {
    switch id {
        case 0u: { return vec2(-1.0, -1.0); }
        case 1u: { return vec2( 1.0, -1.0); }
        case 2u: { return vec2( 1.0,  1.0); }
        case 3u: { return vec2(-1.0, -1.0); }
        case 4u: { return vec2( 1.0,  1.0); }
        default: { return vec2(-1.0,  1.0); }
    }
}

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

fn extra_slice_alpha(position: array<f32, MAX_DIM>) -> f32 {
    var alpha = 1.0;
    let dimension = min(camera.dimension_data.x, MAX_DIM);
    for (var d = 3u; d < dimension; d++) {
        let center = slice_center(d);
        let half = max(slice_width(d) * 0.5, 1e-6);
        let distance = abs(position_component(position, d) - center);
        alpha = min(alpha, 1.0 - smoothstep(half * 0.8, half, distance));
    }
    return alpha;
}

@vertex
fn main(
    input: VertexInput,
    @builtin(vertex_index) corner_id: u32,
    @builtin(instance_index) instance_id: u32,
) -> VertexOutput {
    var output: VertexOutput;
    output.soft = camera.render_data.x;

    var full_position: array<f32, MAX_DIM>;
    full_position[0] = input.position_low.x;
    full_position[1] = input.position_low.y;
    full_position[2] = input.position_low.z;
    full_position[3] = input.position_low.w;
    full_position[4] = input.position_high.x;
    full_position[5] = input.position_high.y;
    full_position[6] = input.position_high.z;
    full_position[7] = input.position_high.w;
    full_position=project_nd(full_position);
    output.slice_alpha = extra_slice_alpha(full_position);

    let is_selected = selection[instance_id] != 0u;
    var radius = camera.particle_size;
    if is_selected {
        output.color = vec3<f32>(1.0);
        radius *= 1.5;
    } else {
        output.color = type_color(input.kind);
    }

    let right = normalize(vec3<f32>(
        camera.view_proj[0][0], camera.view_proj[1][0], camera.view_proj[2][0]
    ));
    let up = normalize(vec3<f32>(
        camera.view_proj[0][1], camera.view_proj[1][1], camera.view_proj[2][1]
    ));
    let corner = corner_offset(corner_id);
    output.offset = corner;

    let position=vec3<f32>(full_position[0],full_position[1],full_position[2]);
    let world = position + right * corner.x * radius + up * corner.y * radius;
    output.clip_position = camera.view_proj * vec4<f32>(world, 1.0);
    return output;
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
