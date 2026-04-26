struct Particle {
    position: vec3<f32>,
    kind:     u32,
    velocity: vec3<f32>,
    prefab_id: i32,
};

struct Params {
    dt:          f32,
    r_max:       f32,
    force_scale: f32,
    friction:    f32,
    beta:        f32,
    bounds:      f32,
    max_speed:   f32,
    type_count:  u32,
    count:       u32,
    wrap:        u32,   // 0 or 1
    _pad0:       u32,
    _pad1:       u32,
};

struct SelectionParams {
    mode: u32,
    _pad0: vec3<u32>,

    rect_min: vec2<f32>,
    rect_max: vec2<f32>,

    brush_center: vec2<f32>,
    brush_radius: f32,

    slice_depth: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform>             params:    Params;
@group(0) @binding(2) var<storage, read>       rules:     array<f32>;
@group(0) @binding(3) var<storage, read_write> selection: array<u32>;
@group(0) @binding(4) var<uniform>             sel_params: SelectionParams;

fn wrap_delta(d: f32, bounds: f32) -> f32 {
    let half = bounds * 0.5;
    if d > half  { return d - bounds; }
    if d < -half { return d + bounds; }
    return d;
}

fn force_law(dn: f32, attr: f32, beta: f32) -> f32 {
    if dn < beta {
        return dn / beta - 1.0;
    }
    return attr * (1.0 - abs((2.0 * dn - 1.0 - beta) / (1.0 - beta)));
}

fn check_selection(pos: vec3<f32>, sel: SelectionParams) -> bool {
    if sel.mode == 0u {
        return false;
    }
    
    if sel.mode == 1u { // Rectangle selection
        let screen_pos = pos.xy;
        return screen_pos.x >= sel.rect_min.x && screen_pos.x <= sel.rect_max.x &&
               screen_pos.y >= sel.rect_min.y && screen_pos.y <= sel.rect_max.y;
    } else if sel.mode == 2u { // Brush selection
        let screen_pos = pos.xy;
        let dist = distance(screen_pos, sel.brush_center);
        return dist <= sel.brush_radius;
    } else if sel.mode == 3u { // Slice selection
        return pos.z <= sel.slice_depth;
    }
    
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count { return; }

    let pi   = particles[i];
    let r2   = params.r_max * params.r_max;
    var f    = vec3<f32>(0.0);

    for (var j = 0u; j < params.count; j++) {
        if j == i { continue; }
        let pj = particles[j];

        var d = pj.position - pi.position;
        if params.wrap != 0u {
            d.x = wrap_delta(d.x, params.bounds);
            d.y = wrap_delta(d.y, params.bounds);
            d.z = wrap_delta(d.z, params.bounds);
        }

        let dist_sq = dot(d, d);
        if dist_sq < 1e-10 || dist_sq > r2 { continue; }

        let dist = sqrt(dist_sq);
        let dn   = dist / params.r_max;
        let attr = rules[pi.kind * params.type_count + pj.kind];
        let a    = force_law(dn, attr, params.beta);
        f += a * (d / dist);
    }

    // Integrate velocity
    let damping = pow(params.friction, params.dt * 60.0);
    var vel = (pi.velocity + f * params.force_scale * params.dt) * damping;

    // Clamp speed
    let spd = length(vel);
    if spd > params.max_speed { vel = vel * (params.max_speed / spd); }

    // Integrate position
    var pos = pi.position + vel * params.dt;

    // Wrap
    if params.wrap != 0u {
        pos = (pos % params.bounds + params.bounds) % params.bounds;
    }

    // Write back
    particles[i].velocity = vel;
    particles[i].position = pos;
    
    // Update selection
    let is_selected = check_selection(pos, sel_params);
    selection[i] = select(0u, 1u, is_selected);
}
