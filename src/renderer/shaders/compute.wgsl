struct Particle {
    position: vec3<f32>,
    kind:     u32,
    velocity: vec3<f32>,
    prefab_id: i32,
};

struct Params {
    dt:                   f32,
    r_max:                f32,
    force_scale:          f32,
    friction:             f32,
    beta:                 f32,
    bounds:               f32,
    max_speed:            f32,
    type_count:           u32,
    count:                u32,
    wrap:                 u32,   // 0 or 1
    reactions_enabled:    u32,
    mix_radius:           f32,
    reaction_probability: f32,
    _pad0:                u32,
    _pad1:                u32,
    _pad2:                u32,
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

@group(0) @binding(0) var<storage, read>  particles_in:  array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform>             params:    Params;
@group(0) @binding(3) var<storage, read>       rules:     array<f32>;
@group(0) @binding(4) var<storage, read_write> selection: array<u32>;
@group(0) @binding(5) var<uniform>             sel_params: SelectionParams;
@group(0) @binding(6) var<storage, read>       reaction_table: array<i32>;
@group(0) @binding(7) var<storage, read>       trace_len_matrix: array<u32>;
@group(0) @binding(8) var<storage, read_write> trace_timers: array<u32>;
@group(0) @binding(9) var<storage, read_write> trace_prev_pos: array<vec3<f32>>;

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

    let pi   = particles_in[i];
    let old_pos = pi.position; // Store old position for trace
    let r2   = params.r_max * params.r_max;
    var f    = vec3<f32>(0.0);

    for (var j = 0u; j < params.count; j++) {
        if j == i { continue; }
        let pj = particles_in[j];

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

    // Reactions
    var final_kind = particles_in[i].kind;
    if params.reactions_enabled != 0u {
        let mix_r_sq = params.mix_radius * params.mix_radius;

        for (var j = 0u; j < params.count; j++) {
            if j == i { continue; }

            var dx = particles_in[j].position.x - pos.x;
            var dy = particles_in[j].position.y - pos.y;
            var dz = particles_in[j].position.z - pos.z;

            if params.wrap != 0u {
                let half = params.bounds * 0.5;
                if dx >  half { dx -= params.bounds; }
                if dx < -half { dx += params.bounds; }
                if dy >  half { dy -= params.bounds; }
                if dy < -half { dy += params.bounds; }
                if dz >  half { dz -= params.bounds; }
                if dz < -half { dz += params.bounds; }
            }

            let dist_sq = dx*dx + dy*dy + dz*dz;
            if dist_sq > mix_r_sq { continue; }

            let ri = particles_in[i].kind;
            let rj = particles_in[j].kind;
            let result = reaction_table[ri * params.type_count + rj];
            if result >= 0 {
                // Use a deterministic per-pair hash as a cheap "random" gate
                let hash = (i * 2654435761u + j * 2246822519u + u32(params.dt * 1000.0)) & 0xFFFFu;
                let threshold = u32(params.reaction_probability * 65535.0);
                if hash < threshold {
                    final_kind = u32(result);
                    
                    // Set trace timer based on trace length matrix
                    let trace_idx = ri * params.type_count + rj;
                    let trace_life = trace_len_matrix[trace_idx];
                    if trace_life > 0u {
                        trace_timers[i] = trace_life;
                        trace_prev_pos[i] = old_pos;
                    }
                }
            }
        }
    }

    // Write back
    particles_out[i] = Particle(
        pos,
        final_kind,
        vel,
        pi.prefab_id
    );
    
    // Update selection
    let is_selected = check_selection(pos, sel_params);
    selection[i] = u32(is_selected);
    
    // Decrement trace timer
    if trace_timers[i] > 0u {
        trace_timers[i] = trace_timers[i] - 1u;
    }
}
