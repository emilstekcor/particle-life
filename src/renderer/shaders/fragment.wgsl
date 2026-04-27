struct FragmentInput {
    @location(0) color: vec3<f32>,
    @location(1) size: f32,
};

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    // For now, just render the color with full opacity
    // The size parameter is passed through but not used for point rendering
    return vec4<f32>(input.color, 1.0);
}
