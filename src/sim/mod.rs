pub mod book;
pub mod physics;

pub use book::Book;

use glam::Vec3;
use rand::Rng;

pub const MAX_RENDER_PARTICLES: usize = 200_000;
pub const MAX_CPU_PHYSICS_PARTICLES: usize = 50_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuStepMode {
    Auto,
    Naive,
    GridExact,
}

// ── Particle struct for CPU-side logic (40 bytes with prefab_local_type)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub kind: u32,
    pub velocity: [f32; 3],
    pub prefab_id: i32,
    pub prefab_local_type: i32,
}

// GPU-optimized particle struct (32 bytes, vec3-aligned)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 3],  // 12 bytes
    pub kind: u32,           // 4 bytes -> 16 bytes total (vec3 alignment)
    pub velocity: [f32; 3],  // 12 bytes
    pub prefab_id: i32,       // 4 bytes -> 32 bytes total
}

impl GpuParticle {
    pub fn from_particle(p: &Particle) -> Self {
        Self {
            position: p.position,
            kind: p.kind,
            velocity: p.velocity,
            prefab_id: p.prefab_id,
        }
    }
}

impl Particle {
    pub fn new(pos: Vec3, kind: u32) -> Self {
        Self {
            position: pos.into(),
            kind,
            velocity: [0.0; 3],
            prefab_id: -1,
            prefab_local_type: -1,
        }
    }
}

// ── SimParams ─────────────────────────────────────────────────────────────────
#[derive(Clone, Debug)]
pub struct SimParams {
    pub type_count: usize,
    pub bounds: f32,
    pub dt: f32,
    pub r_max: f32,
    pub force_scale: f32,
    pub friction: f32,
    pub wrap: bool,
    pub max_speed: f32,
    pub beta: f32,
    pub particle_size: f32,

    // CPU stepping controls
    pub cpu_step_mode: CpuStepMode,
    pub auto_grid_threshold: usize,

    // Reaction system parameters
    pub reactions_enabled: bool,
    pub mix_radius: f32,
    pub reaction_probability: f32,
    pub preserve_particle_count: bool,
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
            max_speed: 0.5,
            beta: 0.30,
            particle_size: 0.02,

            cpu_step_mode: CpuStepMode::Auto,
            auto_grid_threshold: 500,

            reactions_enabled: false,
            mix_radius: 0.15,
            reaction_probability: 0.1,
            preserve_particle_count: false,
        }
    }
}

impl SimParams {
    /// Get the actual r_max scaled by bounds
    pub fn scaled_r_max(&self) -> f32 {
        self.r_max * self.bounds / 20.0
    }
    
    /// Get the actual mix_radius scaled by bounds
    pub fn scaled_mix_radius(&self) -> f32 {
        self.mix_radius * self.bounds / 20.0
    }
    
    /// Get the actual max_speed scaled by bounds
    pub fn scaled_max_speed(&self) -> f32 {
        self.max_speed * self.bounds / 20.0
    }
}

// ── SimState ──────────────────────────────────────────────────────────────────
pub struct SimState {
    pub particles: Vec<Particle>,
    pub params: SimParams,
    pub force_matrix: Vec<f32>,
    pub reaction_table: Vec<i32>,
    pub book: Book,
    pub step_count: u64,
    pub particles_dirty: bool,
    pub params_dirty: bool,
    pub force_matrix_dirty: bool,
    pub next_prefab_instance_id: i32,

    // Scratch buffers for performance (reused each frame)
    pub vel_scratch: Vec<[f32; 3]>,
    pub buckets_scratch: Vec<Vec<usize>>,
    pub reaction_changes_scratch: Vec<(usize, u32)>,

    // Debug/telemetry for later UI
    pub last_step_used_grid: bool,
    pub last_neighbor_checks: u64,
    pub last_grid_res: usize,
    pub last_step_ms: f32,
    pub avg_step_ms: f32,
}

impl SimState {
    pub fn new() -> Self {
        let params = SimParams::default();
        let n = params.type_count;

        let mut state = Self {
            particles: Vec::new(),
            force_matrix: vec![0.0; n * n],
            reaction_table: vec![-1; n * n],
            book: Book::new(),
            step_count: 0,
            particles_dirty: true,
            params_dirty: true,
            force_matrix_dirty: true,
            params,
            next_prefab_instance_id: 1,

            vel_scratch: Vec::new(),
            buckets_scratch: Vec::new(),
            reaction_changes_scratch: Vec::new(),

            last_step_used_grid: false,
            last_neighbor_checks: 0,
            last_grid_res: 1,
            last_step_ms: 0.0,
            avg_step_ms: 0.0,
        };

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

    // Reaction table methods
    pub fn resize_reaction_table(&mut self) {
        let n = self.params.type_count;
        self.reaction_table = vec![-1; n * n];
    }

    pub fn default_reaction_table(&mut self) {
        let n = self.params.type_count;
        for i in 0..n {
            for j in 0..n {
                self.reaction_table[i * n + j] = ((i + j) % n) as i32;
            }
        }
    }

    pub fn rx(&self, i: usize, j: usize) -> i32 {
        self.reaction_table[i * self.params.type_count + j]
    }

    pub fn step(&mut self) {
        crate::sim::physics::cpu_step(self);
        self.step_count += 1;
    }

    #[allow(dead_code)]
    pub fn clear_dirty(&mut self) {
        self.particles_dirty = false;
        self.params_dirty = false;
        self.force_matrix_dirty = false;
    }

    pub fn spawn_random(&mut self, count: usize) {
        use rand::Rng;
        let cap = MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES);
        let remaining = cap.saturating_sub(self.particles.len());
        let spawn_n = count.min(remaining);
        if spawn_n == 0 { 
            log::warn!("Particle cap reached ({})", cap); 
            return; 
        }
        
        let mut rng = rand::thread_rng();
        for _ in 0..spawn_n {
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

    pub fn clear_particles(&mut self) {
        self.particles.clear();
        self.particles_dirty = true;
    }

    pub fn allocate_prefab_instance_id(&mut self) -> i32 {
        let id = self.next_prefab_instance_id;
        self.next_prefab_instance_id += 1;
        id
    }

    pub fn spawn_prefab(&mut self, prefab_index: usize, position: glam::Vec3, prefab_id: i32) {
        if prefab_index >= self.book.prefabs.len() { return; }
        let prefab = &self.book.prefabs[prefab_index];
        
        let cap = MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES);
        let remaining = cap.saturating_sub(self.particles.len());
        
        let mut added = 0;
        for pp in &prefab.particles {
            if added >= remaining {
                log::warn!("Particle cap reached during prefab spawn ({})", cap);
                break;
            }
            let world_pos = position + glam::Vec3::from(pp.relative_position);
            let mut p = Particle::new(world_pos, pp.kind);
            p.prefab_id = prefab_id;
            self.particles.push(p);
            added += 1;
        }
        if added > 0 {
            self.particles_dirty = true;
        }
    }

    pub fn delete_particles(&mut self, indices: &[usize]) {
        if indices.is_empty() { return; }
        let to_delete: std::collections::HashSet<usize> = indices.iter().copied().collect();
        let mut write = 0;
        for read in 0..self.particles.len() {
            if !to_delete.contains(&read) {
                self.particles.swap(write, read);
                write += 1;
            }
        }
        self.particles.truncate(write);
        self.particles_dirty = true;
    }

    pub fn duplicate_particles(&mut self, indices: &[usize]) {
        if indices.is_empty() { return; }
        
        let cap = MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES);
        let remaining = cap.saturating_sub(self.particles.len());
        let to_duplicate = indices.len().min(remaining);
        
        if to_duplicate == 0 {
            log::warn!("Particle cap reached during duplication ({})", cap);
            return;
        }
        
        let orig_len = self.particles.len();
        let mut rng = rand::thread_rng();
        for &idx in indices.iter().take(to_duplicate) {
            if idx < orig_len {
                let orig = self.particles[idx];
                let mut dup = orig;
                let offset = glam::Vec3::new(
                    rng.gen_range(-0.02..0.02),
                    rng.gen_range(-0.02..0.02),
                    rng.gen_range(-0.02..0.02),
                );
                dup.position = (glam::Vec3::from(orig.position) + offset).into();
                dup.prefab_id = -1;
                dup.prefab_local_type = -1;
                self.particles.push(dup);
            }
        }
        self.particles_dirty = true;
    }

    pub fn assign_type_to_particles(&mut self, indices: &[usize], kind: u32) {
        let max_kind = self.params.type_count.saturating_sub(1) as u32;
        let kind = kind.min(max_kind);
        for &idx in indices {
            if idx < self.particles.len() {
                self.particles[idx].kind = kind;
            }
        }
        self.particles_dirty = true;
    }

    #[allow(dead_code)]
    pub fn move_particles(&mut self, indices: &[usize], delta: glam::Vec3) {
        if indices.is_empty() { return; }
        for &idx in indices {
            if idx < self.particles.len() {
                let pos = &mut self.particles[idx].position;
                *pos = (glam::Vec3::from(*pos) + delta).into();
            }
        }
        self.particles_dirty = true;
    }

    pub fn scale_velocities(&mut self, factor: f32) {
        for p in &mut self.particles {
            p.velocity[0] *= factor;
            p.velocity[1] *= factor;
            p.velocity[2] *= factor;
        }
        self.particles_dirty = true;
    }

    pub fn set_type_count(&mut self, new_count: usize) {
        if new_count == self.params.type_count { return; }
        let old_count = self.params.type_count;
        self.params.type_count = new_count;

        let mut new_matrix = vec![0.0_f32; new_count * new_count];
        for i in 0..old_count.min(new_count) {
            for j in 0..old_count.min(new_count) {
                new_matrix[i * new_count + j] = self.force_matrix[i * old_count + j];
            }
        }
        self.force_matrix = new_matrix;
        self.force_matrix_dirty = true;

        // Resize reaction table
        let mut new_reaction_table = vec![-1i32; new_count * new_count];
        for i in 0..old_count.min(new_count) {
            for j in 0..old_count.min(new_count) {
                new_reaction_table[i * new_count + j] = self.reaction_table[i * old_count + j];
            }
        }
        self.reaction_table = new_reaction_table;

        for p in &mut self.particles {
            if p.kind >= new_count as u32 {
                p.kind = (new_count - 1) as u32;
            }
        }
        self.particles_dirty = true;
        self.params_dirty = true;
    }
}