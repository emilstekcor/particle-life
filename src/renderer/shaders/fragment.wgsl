struct FragmentInput {
    @location(0) color: vec3<f32>,
    @location(1) offset: vec2<f32>,
    @location(2) soft: f32,
    @location(3) slice_alpha: f32,
};

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    // offset spans -1..1 across the billboard, so its length is the normalized
    // distance from centre. Discarding past 1.0 turns the quad into a disc;
    // without it every particle renders as a square.
    let d = length(input.offset);
    if (d > 1.0 || input.slice_alpha <= 0.0) {
        discard;
    }

    // Hard mode writes alpha 1.0, which makes the alpha-blended pipeline behave
    // exactly like REPLACE: fully order-independent, so overlapping particles
    // can't produce frame-to-frame flicker that isn't in the physics. That
    // matters when studying period-2 structures. Soft mode feathers the last
    // pixel using fwidth, which stays one pixel wide at any size or zoom.
    if (input.soft < 0.5) {
        return vec4<f32>(input.color, input.slice_alpha);
    }

    let edge = fwidth(d) * 1.5;
    let alpha = 1.0 - smoothstep(1.0 - edge, 1.0, d);
    return vec4<f32>(input.color, alpha * input.slice_alpha);
}
