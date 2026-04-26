struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>) -> VsOut {
    var out: VsOut;
    out.pos = vec4<f32>(position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // dark transparent overlay
    return vec4<f32>(0.02, 0.02, 0.02, 0.08);
}
