struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) kind:     u32,
    @builtin(instance_index) instance_id: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct CameraUniform {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<storage, read> selection: array<u32>;

fn type_color(kind: u32) -> vec3<f32> {
    let index = kind % 8u;
    if (index == 0u) { return vec3(0.96, 0.36, 0.36); }
    if (index == 1u) { return vec3(0.36, 0.76, 0.96); }
    if (index == 2u) { return vec3(0.56, 0.96, 0.36); }
    if (index == 3u) { return vec3(0.96, 0.76, 0.26); }
    if (index == 4u) { return vec3(0.86, 0.46, 0.96); }
    if (index == 5u) { return vec3(0.96, 0.56, 0.26); }
    if (index == 6u) { return vec3(0.36, 0.96, 0.76); }
    return vec3(0.96, 0.76, 0.86);
}

@vertex
fn main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = camera.view_proj * vec4<f32>(input.position, 1.0);
    
    // Check if this particle is selected
    let is_selected = selection[input.instance_id] != 0u;
    
    if (is_selected) {
        output.color = vec3<f32>(1.0, 1.0, 1.0); // White for selected
    } else {
        output.color = type_color(input.kind);
    }
    
    return output;
}
