pub mod book;
pub mod physics;

pub use book::Book;

use glam::Vec3;
use rand::Rng;
use std::collections::HashMap;

pub const MIN_DIM: usize = 3;
pub const MAX_DIM: usize = 8;
pub const MAX_TYPES: usize = 32;
pub const MAX_RENDER_PARTICLES: usize = 200_000;
pub const MAX_CPU_PHYSICS_PARTICLES: usize = 50_000;

pub type VecNd = [f32; MAX_DIM];
pub type CellCoord = [usize; MAX_DIM];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CpuStepMode {
    Auto,
    Naive,
    GridExact,
}

// CPU-side particle. Fixed-width arrays avoid reallocating every particle when
// the active runtime dimension changes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, serde::Serialize, serde::Deserialize,
)]
pub struct Particle {
    pub position: VecNd,
    pub kind: u32,
    pub velocity: VecNd,
    pub prefab_id: i32,
    pub prefab_local_type: i32,
    _pad: u32,
}

// GPU particle layout. The explicit tail pad makes the 80-byte stride a
// multiple of 16 and exactly matches the WGSL storage structs.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable, serde::Serialize, serde::Deserialize,
)]
pub struct GpuParticle {
    pub position: VecNd,
    pub kind: u32,
    pub velocity: VecNd,
    pub prefab_id: i32,
    _pad: [u32; 2],
}

impl GpuParticle {
    pub fn from_particle(p: &Particle) -> Self {
        Self {
            position: p.position,
            kind: p.kind,
            velocity: p.velocity,
            prefab_id: p.prefab_id,
            _pad: [0; 2],
        }
    }
}

impl Particle {
    pub fn new(pos: Vec3, kind: u32) -> Self {
        let mut position = [0.0; MAX_DIM];
        position[..3].copy_from_slice(&pos.to_array());
        Self {
            position,
            kind,
            velocity: [0.0; MAX_DIM],
            prefab_id: -1,
            prefab_local_type: -1,
            _pad: 0,
        }
    }
}

// ── SimParams ─────────────────────────────────────────────────────────────────
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SimParams {
    pub gpu_grid: bool,
    pub type_count: usize,
    pub dimension: usize,
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
            gpu_grid: false,
            type_count: 4,
            dimension: MIN_DIM,
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
    pub reaction_table_dirty: bool,
    pub trace_len_matrix: Vec<u32>,
    pub trace_timers: Vec<u32>,
    pub trace_len_matrix_dirty: bool,
    pub book: Book,
    pub seed: u64,
    pub random_counter: u64,
    pub step_count: u64,
    pub particles_dirty: bool,
    /// Set only when `particles` was rebuilt wholesale (spawn, clear, load) and
    /// the GPU genuinely must be overwritten.
    ///
    /// `particles_dirty` alone is NOT sufficient to justify an upload: under GPU
    /// physics the CPU mirror is intentionally stale, so uploading it rewinds
    /// the whole simulation to whenever GPU physics was switched on.
    pub particles_replaced: bool,
    pub params_dirty: bool,
    pub force_matrix_dirty: bool,
    pub next_prefab_instance_id: i32,

    // Scratch buffers for performance (reused each frame)
    pub vel_scratch: Vec<VecNd>,
    pub buckets_scratch: HashMap<CellCoord, Vec<usize>>,
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
            reaction_table_dirty: true,
            trace_len_matrix: vec![0u32; n * n],
            trace_timers: Vec::new(),
            trace_len_matrix_dirty: true,
            book: Book::new(),
            seed: 1,
            random_counter: 0,
            step_count: 0,
            particles_dirty: true,
            particles_replaced: true,
            params_dirty: true,
            force_matrix_dirty: true,
            params,
            next_prefab_instance_id: 1,

            vel_scratch: Vec::new(),
            buckets_scratch: HashMap::new(),
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

    pub fn next_rng(&mut self) -> rand::rngs::StdRng {
        use rand::SeedableRng;
        let seed = self
            .seed
            .wrapping_add(self.random_counter.wrapping_mul(0x9E3779B97F4A7C15));
        self.random_counter = self.random_counter.wrapping_add(1);
        rand::rngs::StdRng::seed_from_u64(seed)
    }
    pub fn randomize_rules(&mut self) {
        use rand::Rng;
        let mut rng = self.next_rng();
        let n = self.params.type_count;
        self.force_matrix = (0..n * n)
            .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
            .collect();
        self.force_matrix_dirty = true;
    }
    pub fn respawn_mix(&mut self, mix: &[usize]) -> usize {
        use rand::Rng;
        let cap = MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES);
        let n = self.params.type_count;

        self.particles.clear();
        let mut rng = self.next_rng();
        let mut total = 0usize;

        for (kind, &count) in mix.iter().enumerate().take(n) {
            for _ in 0..count {
                if total >= cap {
                    log::warn!("Particle cap reached during respawn ({cap})");
                    self.particles_dirty = true;
                    self.particles_replaced = true;
                    return total;
                }
                let mut particle = Particle::new(Vec3::ZERO, kind as u32);
                for d in 0..self.params.dimension {
                    particle.position[d] = rng.gen_range(0.0..self.params.bounds);
                }
                self.particles.push(particle);
                total += 1;
            }
        }

        self.particles_dirty = true;
        self.particles_replaced = true;
        total
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
        self.reaction_table_dirty = true;
    }

    pub fn default_reaction_table(&mut self) {
        self.edit_reaction_table(|reaction_table, n| {
            for i in 0..n {
                for j in 0..n {
                    reaction_table[i * n + j] = ((i + j) % n) as i32;
                }
            }
        });
    }

    pub fn rx(&self, i: usize, j: usize) -> i32 {
        self.reaction_table[i * self.params.type_count + j]
    }

    pub fn set_reaction(&mut self, a: usize, b: usize, value: i32) {
        let n = self.params.type_count;
        if a >= n || b >= n {
            return;
        }
        let clamped = value.clamp(-1, (n.saturating_sub(1)) as i32);
        let idx = a * n + b;
        if self.reaction_table[idx] != clamped {
            self.reaction_table[idx] = clamped;
            self.reaction_table_dirty = true;
            log::debug!("reaction table marked dirty");
        }
    }

    pub fn edit_reaction_table(&mut self, f: impl FnOnce(&mut [i32], usize)) {
        let n = self.params.type_count;
        f(&mut self.reaction_table, n);
        self.reaction_table_dirty = true;
        log::debug!("reaction table marked dirty");
    }

    pub fn set_reactions_enabled(&mut self, v: bool) {
        if self.params.reactions_enabled != v {
            self.params.reactions_enabled = v;
            self.params_dirty = true;
        }
    }

    pub fn set_mix_radius(&mut self, v: f32) {
        let v = v.max(0.0);
        if self.params.mix_radius != v {
            self.params.mix_radius = v;
            self.params_dirty = true;
        }
    }

    pub fn set_reaction_probability(&mut self, v: f32) {
        let v = v.clamp(0.0, 1.0);
        if self.params.reaction_probability != v {
            self.params.reaction_probability = v;
            self.params_dirty = true;
        }
    }

    pub fn set_preserve_particle_count(&mut self, v: bool) {
        if self.params.preserve_particle_count != v {
            self.params.preserve_particle_count = v;
            self.params_dirty = true;
        }
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
        self.reaction_table_dirty = false;
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

        let mut rng = self.next_rng();
        for _ in 0..spawn_n {
            let mut pos = [0.0; MAX_DIM];
            for coord in pos.iter_mut().take(self.params.dimension) {
                *coord = rng.gen_range(0.0..self.params.bounds);
            }
            let kind = rng.gen_range(0..self.params.type_count) as u32;
            let mut particle = Particle::new(Vec3::ZERO, kind);
            particle.position = pos;
            self.particles.push(particle);
        }
        self.particles_dirty = true;
    }

    pub fn allocate_prefab_instance_id(&mut self) -> i32 {
        let id = self.next_prefab_instance_id;
        self.next_prefab_instance_id += 1;
        id
    }

    pub fn spawn_prefab(&mut self, prefab_index: usize, position: glam::Vec3, prefab_id: i32) {
        if prefab_index >= self.book.prefabs.len() {
            return;
        }
        let prefab = &self.book.prefabs[prefab_index];

        let cap = MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES);
        let remaining = cap.saturating_sub(self.particles.len());
        // A prefab remembers the type_count it was recorded under. If the live
        // simulation has since been set to fewer types (Controls > Physics),
        // its particles' `kind` values can name types that no longer exist —
        // clamp so a spawn never indexes force_matrix/reaction_table out of
        // bounds and panics.
        let max_kind = self.params.type_count.saturating_sub(1) as u32;

        let mut added = 0;
        for pp in &prefab.particles {
            if added >= remaining {
                log::warn!("Particle cap reached during prefab spawn ({})", cap);
                break;
            }
            let mut p = Particle::new(position, pp.kind.min(max_kind));
            for d in 0..self.params.dimension {
                let base = if d < 3 {
                    position.to_array()[d]
                } else {
                    self.params.bounds * 0.5
                };
                p.position[d] = base + pp.relative_position.get(d).copied().unwrap_or(0.0);
                if self.params.wrap {
                    p.position[d] = p.position[d].rem_euclid(self.params.bounds);
                }
                p.velocity[d] = pp.velocity.get(d).copied().unwrap_or(0.0);
            }
            p.prefab_id = prefab_id;
            self.particles.push(p);
            added += 1;
        }
        if added > 0 {
            self.particles_dirty = true;
        }
    }

    pub fn delete_particles(&mut self, indices: &[usize]) {
        if indices.is_empty() {
            return;
        }
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
        if indices.is_empty() {
            return;
        }

        let cap = MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES);
        let remaining = cap.saturating_sub(self.particles.len());
        let to_duplicate = indices.len().min(remaining);

        if to_duplicate == 0 {
            log::warn!("Particle cap reached during duplication ({})", cap);
            return;
        }

        let orig_len = self.particles.len();
        let mut rng = self.next_rng();
        for &idx in indices.iter().take(to_duplicate) {
            if idx < orig_len {
                let orig = self.particles[idx];
                let mut dup = orig;
                for d in 0..self.params.dimension {
                    dup.position[d] += rng.gen_range(-0.02..0.02);
                }
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
        if indices.is_empty() {
            return;
        }
        for &idx in indices {
            if idx < self.particles.len() {
                let pos = &mut self.particles[idx].position;
                pos[0] += delta.x;
                pos[1] += delta.y;
                pos[2] += delta.z;
            }
        }
        self.particles_dirty = true;
    }

    pub fn scale_velocities(&mut self, factor: f32) {
        for p in &mut self.particles {
            for velocity in p.velocity.iter_mut().take(self.params.dimension) {
                *velocity *= factor;
            }
        }
        self.particles_dirty = true;
    }

    /// Change the number of simulated dimensions without reallocating particle
    /// storage. Newly enabled axes are spread through the domain; disabled axes
    /// are cleared so returning to 3D exactly removes their contribution.
    pub fn set_dimension(&mut self, new_dimension: usize) {
        let new_dimension = new_dimension.clamp(MIN_DIM, MAX_DIM);
        let old_dimension = self.params.dimension;
        if new_dimension == old_dimension {
            return;
        }

        let mut rng = self.next_rng();
        for particle in &mut self.particles {
            if new_dimension > old_dimension {
                for d in old_dimension..new_dimension {
                    particle.position[d] = rng.gen_range(0.0..self.params.bounds);
                    particle.velocity[d] = 0.0;
                }
            } else {
                for d in new_dimension..MAX_DIM {
                    particle.position[d] = 0.0;
                    particle.velocity[d] = 0.0;
                }
            }
        }

        self.params.dimension = new_dimension;
        self.params_dirty = true;
        self.particles_dirty = true;
        self.particles_replaced = true;
        self.step_count = 0;
    }

    pub fn set_type_count(&mut self, new_count: usize) {
        let new_count = new_count.clamp(1, MAX_TYPES);
        if new_count == self.params.type_count {
            return;
        }
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
                let value = self.reaction_table[i * old_count + j];
                new_reaction_table[i * new_count + j] =
                    if value >= new_count as i32 { -1 } else { value };
            }
        }
        self.reaction_table = new_reaction_table;
        self.reaction_table_dirty = true;

        // Resize trace length matrix
        let mut new_trace_len_matrix = vec![0u32; new_count * new_count];
        for i in 0..old_count.min(new_count) {
            for j in 0..old_count.min(new_count) {
                new_trace_len_matrix[i * new_count + j] = self.trace_len_matrix[i * old_count + j];
            }
        }
        self.trace_len_matrix = new_trace_len_matrix;
        self.trace_len_matrix_dirty = true;

        for p in &mut self.particles {
            if p.kind >= new_count as u32 {
                p.kind = (new_count - 1) as u32;
            }
        }
        // NOTE: deliberately does NOT set particles_replaced. Only `kind` was
        // touched; positions and velocities in this array may be a stale mirror
        // of GPU state. The renderer re-syncs from the GPU before honouring
        // this flag, so dragging the types slider no longer rewinds the sim.
        self.particles_dirty = true;
        self.params_dirty = true;
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn particle_layouts_match_gpu_contract() {
        assert_eq!(std::mem::size_of::<Particle>(), 80);
        assert_eq!(std::mem::size_of::<GpuParticle>(), 80);
    }
}
