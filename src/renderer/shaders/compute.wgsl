const MAX_DIM: u32 = 8u;

struct Particle {
    position: array<f32, MAX_DIM>,
    kind: u32,
    velocity: array<f32, MAX_DIM>,
    prefab_id: i32,
    _pad: vec2<u32>,
};

struct Params {
    dt: f32,
    r_max: f32,
    force_scale: f32,
    friction: f32,
    beta: f32,
    bounds: f32,
    max_speed: f32,
    type_count: u32,
    count: u32,
    wrap: u32,
    reactions_enabled: u32,
    mix_radius: f32,
    reaction_probability: f32,
    frame: u32,
    dimension: u32,
    grid_res: u32,
};

@group(0) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> rules: array<f32>;
@group(0) @binding(6) var<storage, read> reaction_table: array<i32>;
@group(0) @binding(7) var<storage, read> trace_len_matrix: array<u32>;
@group(0) @binding(8) var<storage, read_write> trace_timers: array<u32>;
@group(0) @binding(9) var<storage, read_write> grid_next: array<u32>;
@group(0) @binding(4) var<storage, read_write> grid_heads: array<atomic<u32>>;

fn wrap_delta(value: f32, bounds: f32) -> f32 {
    let half = bounds * 0.5;
    if value > half { return value - bounds; }
    if value < -half { return value + bounds; }
    return value;
}

fn hash_u32(value: u32) -> u32 {
    var hash = value;
    hash ^= hash >> 16u;
    hash *= 0x7feb352du;
    hash ^= hash >> 15u;
    hash *= 0x846ca68bu;
    hash ^= hash >> 16u;
    return hash;
}

fn force_law(distance: f32, attraction: f32, beta: f32) -> f32 {
    if distance < beta {
        return distance / beta - 1.0;
    }
    return attraction * (1.0 - abs((2.0 * distance - 1.0 - beta) / (1.0 - beta)));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count { return; }

    let dimension = min(params.dimension, MAX_DIM);
    let particle = particles_in[i];
    var particle_position = particle.position;
    let radius_sq = params.r_max * params.r_max;
    var force: array<f32, MAX_DIM>;

    var iter=new_iterator(particle_position);
    loop {let j=next_neighbor(&iter);if j==0xffffffffu {break;}
        if j == i { continue; }
        let other = particles_in[j];
        var other_position = other.position;
        var delta: array<f32, MAX_DIM>;
        var distance_sq = 0.0;
        for (var d = 0u; d < dimension; d++) {
            var component = other_position[d] - particle_position[d];
            if params.wrap != 0u {
                component = wrap_delta(component, params.bounds);
            }
            delta[d] = component;
            distance_sq += component * component;
        }
        if distance_sq < 1e-10 || distance_sq > radius_sq { continue; }

        let distance = sqrt(distance_sq);
        let normalized = distance / params.r_max;
        let attraction = rules[particle.kind * params.type_count + other.kind];
        let magnitude = force_law(normalized, attraction, params.beta);
        for (var d = 0u; d < dimension; d++) {
            force[d] += magnitude * delta[d] / distance;
        }
    }

    let damping = pow(params.friction, params.dt * 60.0);
    var velocity = particle.velocity;
    var speed_sq = 0.0;
    for (var d = 0u; d < dimension; d++) {
        velocity[d] =
            (velocity[d] + force[d] * params.force_scale * params.dt) * damping;
        speed_sq += velocity[d] * velocity[d];
    }
    let speed = sqrt(speed_sq);
    if speed > params.max_speed {
        let scale = params.max_speed / speed;
        for (var d = 0u; d < dimension; d++) {
            velocity[d] *= scale;
        }
    }

    var position = particle.position;
    for (var d = 0u; d < dimension; d++) {
        position[d] += velocity[d] * params.dt;
        if params.wrap != 0u {
            position[d] = (position[d] % params.bounds + params.bounds) % params.bounds;
        }
    }

    var result=particle;result.position=position;result.velocity=velocity;particles_out[i]=result;
}
fn gate(i:u32,j:u32)->bool {
 let h=hash_u32(i*2654435761u+j*2246822519u+params.frame*2246822519u)&65535u;
 return params.reaction_probability>=1.0 || h<u32(clamp(params.reaction_probability,0.0,1.0)*65536.0);
}
@compute @workgroup_size(64)
fn react(@builtin(global_invocation_id) gid:vec3<u32>) {
 let i=gid.x;if i>=params.count {return;}
 var particle=particles_in[i];var position=particle.position;
 var final_kind=particle.kind;var winning_neighbor=0u;var reacted=false;
 var timer=trace_timers[i];if timer>0u {timer-=1u;}
 if params.reactions_enabled!=0u {
  var iter=new_iterator(position);
  loop {
   let j=next_neighbor(&iter);if j==0xffffffffu {break;}if i==j {continue;}
   let other=particles_in[j];var other_position=other.position;var distance_sq=0.0;
   for(var d=0u;d<params.dimension;d++){var delta=other_position[d]-position[d];if params.wrap!=0u {delta=wrap_delta(delta,params.bounds);}distance_sq+=delta*delta;}
   if distance_sq>params.mix_radius*params.mix_radius {continue;}
   let ab=particle.kind*params.type_count+other.kind;let ba=other.kind*params.type_count+particle.kind;
   if reaction_table[ab]>=0 && gate(i,j) {
    if !reacted || j>winning_neighbor {final_kind=u32(reaction_table[ab]);winning_neighbor=j;reacted=true;}
    timer=max(timer,trace_len_matrix[ab]);
   }
   if reaction_table[ba]>=0 && gate(j,i) {timer=max(timer,trace_len_matrix[ba]);}
  }
 }
 particle.kind=final_kind;particles_out[i]=particle;trace_timers[i]=timer;
}
// Conservative XYZ broad phase; the narrow phase still checks every active axis.
fn grid_cell(pos:array<f32,8>)->vec3<i32> {
 var p=vec3<f32>(pos[0],pos[1],pos[2]);
 if params.wrap!=0u {p=(p%params.bounds+params.bounds)%params.bounds;}
 return vec3<i32>(clamp(floor(p/params.bounds*f32(params.grid_res)),vec3<f32>(0.0),vec3<f32>(f32(params.grid_res)-1.0)));
}
fn cell_index(cell:vec3<i32>)->u32 {return u32(cell.x)+u32(cell.y)*params.grid_res+u32(cell.z)*params.grid_res*params.grid_res;}
@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id:vec3<u32>){if id.x<params.grid_res*params.grid_res*params.grid_res{atomicStore(&grid_heads[id.x],0xffffffffu);}}
@compute @workgroup_size(64)
fn build_grid(@builtin(global_invocation_id) id:vec3<u32>){if id.x>=params.count{return;}let cell=cell_index(grid_cell(particles_in[id.x].position));grid_next[id.x]=atomicExchange(&grid_heads[cell],id.x);}
struct NeighborIterator {center:vec3<i32>,slot:u32,cursor:u32,linear:u32,}
fn new_iterator(position:array<f32,8>)->NeighborIterator {var it:NeighborIterator;it.slot=0u;it.cursor=0xffffffffu;it.linear=0u;if params.grid_res>=3u{it.center=grid_cell(position);}return it;}
fn next_neighbor(it:ptr<function,NeighborIterator>)->u32 {
 if params.grid_res<3u {let index=(*it).linear;if index>=params.count{return 0xffffffffu;}(*it).linear=index+1u;return index;}
 loop {
  if (*it).cursor!=0xffffffffu {let index=(*it).cursor;(*it).cursor=grid_next[index];return index;}
  if (*it).slot>=27u{return 0xffffffffu;}
  let s=(*it).slot;(*it).slot=s+1u;
  var cell=(*it).center+vec3<i32>(i32(s%3u)-1,i32((s/3u)%3u)-1,i32(s/9u)-1);let res=i32(params.grid_res);
  if params.wrap!=0u {cell=(cell%res+res)%res;}else{if any(cell<vec3<i32>(0))||any(cell>=vec3<i32>(res)){continue;}}
  (*it).cursor=atomicLoad(&grid_heads[cell_index(cell)]);
 }
 return 0xffffffffu;
}
