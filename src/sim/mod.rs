pub mod world;
pub mod book;
pub mod physics;

pub use world::*;
pub use book::Book;

use glam::Vec3;

// ── Particle ──────────────────────────────────────────────────────────────────
// Matches the GPU buffer layout exactly (std430).
// 32 bytes: pos(12) + vel(12) + type(4) + prefab_id(4)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub kind: u32,           // particle type index (0..type_count)
    pub velocity: [f32; 3],
    pub prefab_id: i32,      // -1 = wild particle, >=0 = creature instance id
}

impl Particle {
    pub fn new(pos: Vec3, kind: u32) -> Self {
        Self {
            position: pos.into(),
            kind,
            velocity: [0.0; 3],
            prefab_id: -1,
        }
    }
}

// ── SimParams ─────────────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct SimParams {
    pub type_count: usize,
    pub bounds: f32,          // cube side length (world is [0, bounds]³)
    pub dt: f32,
    pub r_max: f32,           // interaction radius (fraction of bounds)
    pub force_scale: f32,
    pub friction: f32,        // damping per second (0 = no damping, 1 = full stop)
    pub wrap: bool,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            type_count: 4,
            bounds: 1.0,
            dt: 0.016,
            r_max: 0.15,
            force_scale: 1.0,
            friction: 0.1,
            wrap: true,
        }
    }
}

// ── SimState ──────────────────────────────────────────────────────────────────
pub struct SimState {
    pub particles: Vec<Particle>,
    pub params: SimParams,

    // Force matrix: [type_count × type_count] attraction values in [-1, 1]
    // force_matrix[a * type_count + b] = force that type a feels toward type b
    pub force_matrix: Vec<f32>,

    // Creature library loaded from / saved to book.json
    pub book: Book,

    // Increments each step — useful for UI display
    pub step_count: u64,

    // Dirty flags so the GPU knows when to re-upload
    pub particles_dirty: bool,
    pub params_dirty: bool,
    pub force_matrix_dirty: bool,
}

impl SimState {
    pub fn new() -> Self {
        let params = SimParams::default();
        let n = params.type_count;

        let mut state = Self {
            particles: Vec::new(),
            force_matrix: vec![0.0; n * n],
            book: Book::new(),
            step_count: 0,
            particles_dirty: true,
            params_dirty: true,
            force_matrix_dirty: true,
            params,
        };

        // Seed with a default random rule set and some particles
        state.randomize_rules();
        state.spawn_random(512);
        state
    }

    pub fn randomize_rules(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = self.params.type_count;
        self.force_matrix = (0..n * n)
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();
        self.force_matrix_dirty = true;
    }

    pub fn get_rule(&self, a: usize, b: usize) -> f32 {
        self.force_matrix[a * self.params.type_count + b]
    }

    pub fn set_rule(&mut self, a: usize, b: usize, v: f32) {
        self.force_matrix[a * self.params.type_count + b] = v.clamp(-1.0, 1.0);
        self.force_matrix_dirty = true;
    }

    /// Advance one physics tick (CPU path — GPU path called from renderer)
    pub fn step(&mut self) {
        physics::cpu_step(self);
        self.step_count += 1;
        self.particles_dirty = true;
    }

    pub fn clear_dirty(&mut self) {
        self.particles_dirty = false;
        self.params_dirty = false;
        self.force_matrix_dirty = false;
    }

    /// Spawn random particles in the world
    pub fn spawn_random(&mut self, count: usize) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for _ in 0..count {
            let pos = glam::Vec3::new(
                rng.gen_range(0.0..self.params.bounds),
                rng.gen_range(0.0..self.params.bounds),
                rng.gen_range(0.0..self.params.bounds),
            );
            let kind = rng.gen_range(0..self.params.type_count) as u32;
            self.particles.push(Particle::new(pos, kind));
        }
        self.particles_dirty = true;
    }

    /// Clear all particles
    pub fn clear_particles(&mut self) {
        self.particles.clear();
        self.particles_dirty = true;
    }

    /// Spawn a prefab from the book at a specific position
    pub fn spawn_prefab(&mut self, prefab_index: usize, position: glam::Vec3, prefab_id: i32) {
        if prefab_index >= self.book.prefabs.len() {
            return;
        }
        
        let prefab = &self.book.prefabs[prefab_index];
        
        for prefab_particle in &prefab.particles {
            let world_pos = position + glam::Vec3::from(prefab_particle.relative_position);
            let mut particle = Particle::new(world_pos, prefab_particle.kind);
            particle.prefab_id = prefab_id;
            self.particles.push(particle);
        }
        
        self.particles_dirty = true;
    }
}
