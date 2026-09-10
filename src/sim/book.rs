use serde::{Deserialize, Serialize};
use std::fs;

// ── A single particle's data within a prefab ──────────────────────────────────
// Positions are relative to the prefab's center of mass
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefabParticle {
    /// Variable-length on disk for backward compatibility with existing 3D
    /// books while allowing new prefabs to retain every active dimension.
    pub relative_position: Vec<f32>,
    pub kind: u32,
    #[serde(default)]
    pub velocity: Vec<f32>,
}

// ── A saved creature ──────────────────────────────────────────────────────────
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prefab {
    pub name: String,
    pub particles: Vec<PrefabParticle>,

    // The force matrix snapshot at the time this creature was recorded.
    // This is important — the creature only "works" with these rules.
    pub force_matrix: Vec<f32>,
    pub type_count: usize,
    #[serde(default = "default_dimension")]
    pub dimension: usize,

    // Metadata
    pub particle_count: usize,
    pub notes: String,
    #[serde(default)]
    pub params: Option<crate::sim::SimParams>,
    #[serde(default)]
    pub gpu: Option<bool>,
    #[serde(default)]
    pub reactions: Vec<i32>,
    #[serde(default)]
    pub behavior_version: u32,
}

impl Prefab {
    pub fn new(name: String, force_matrix: Vec<f32>, type_count: usize, dimension: usize) -> Self {
        Self {
            name,
            particles: Vec::new(),
            force_matrix,
            type_count,
            dimension,
            particle_count: 0,
            notes: String::new(),
            params: None,
            gpu: None,
            reactions: Vec::new(),
            behavior_version: 1,
        }
    }
}

fn default_dimension() -> usize {
    crate::sim::MIN_DIM
}

// ── The Book — the full creature library ─────────────────────────────────────
#[derive(Debug, Default)]
pub struct Book {
    pub prefabs: Vec<Prefab>,
    pub path: String,
    dirty: bool,
    retry_after: Option<std::time::Instant>,
}

impl Book {
    pub fn new() -> Self {
        Self {
            prefabs: Vec::new(),
            path: "book.json".to_string(),
            dirty: false,
            retry_after: None,
        }
    }

    /// Load prefabs from book.json. Silent no-op if file doesn't exist yet.
    pub fn load_from_file(&mut self, path: &str) {
        self.path = path.to_string();
        match fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str::<Vec<Prefab>>(&contents) {
                Ok(prefabs) => {
                    if prefabs.iter().any(|p| !p.valid()) {
                        log::error!("Invalid creature data in {}; retaining book", path);
                        return;
                    }
                    self.prefabs = prefabs;
                    log::info!("Loaded {} creature(s) from {}", self.prefabs.len(), path);
                }
                Err(e) => log::warn!("Failed to parse {}: {}", path, e),
            },
            Err(_) => {
                // File doesn't exist yet — that's fine, we'll create it on first save
                log::info!("No {} found, starting with empty book", path);
            }
        }
    }

    /// Save all prefabs to book.json
    pub fn save_to_file(&self) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(&self.prefabs).map_err(|e| e.to_string())?;
        crate::session::atomic_write(std::path::Path::new(&self.path), &bytes)
    }
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        self.retry_after = None;
    }
    pub fn flush_if_dirty(&mut self) -> Option<Result<(), String>> {
        if !self.dirty
            || self
                .retry_after
                .map(|t| std::time::Instant::now() < t)
                .unwrap_or(false)
        {
            return None;
        }
        let result = self.save_to_file();
        if result.is_ok() {
            self.dirty = false;
            self.retry_after = None;
        } else {
            self.retry_after = Some(std::time::Instant::now() + std::time::Duration::from_secs(3));
        }
        Some(result)
    }
    pub fn add_prefab(&mut self, prefab: Prefab) {
        self.prefabs.push(prefab);
        self.mark_dirty();
    }

    pub fn remove_prefab(&mut self, index: usize) {
        if index < self.prefabs.len() {
            self.prefabs.remove(index);
            self.mark_dirty();
        }
    }
}

impl Prefab {
    pub fn valid(&self) -> bool {
        let n = self.type_count;
        (1..=crate::sim::MAX_TYPES).contains(&n)
            && (3..=8).contains(&self.dimension)
            && self.particles.len() <= crate::sim::MAX_CPU_PHYSICS_PARTICLES
            && self.force_matrix.len() == n * n
            && self.force_matrix.iter().all(|x| x.is_finite())
            && (self.reactions.is_empty()
                || (self.reactions.len() == n * n
                    && self.reactions.iter().all(|&x| x >= -1 && x < n as i32)))
            && self.particles.iter().all(|p| {
                p.kind < n as u32
                    && p.relative_position.len() <= 8
                    && p.velocity.len() <= 8
                    && p.relative_position
                        .iter()
                        .chain(&p.velocity)
                        .all(|x| x.is_finite())
            })
            && self
                .params
                .as_ref()
                .map(|p| {
                    p.type_count == n
                        && p.dimension == self.dimension
                        && p.bounds.is_finite()
                        && p.bounds > 0.0
                        && p.dt > 0.0
                        && p.dt.is_finite()
                        && p.beta > 0.0
                        && p.beta < 1.0
                        && (0.0..=1.0).contains(&p.friction)
                        && (0.0..=1.0).contains(&p.reaction_probability)
                        && [
                            p.r_max,
                            p.mix_radius,
                            p.max_speed,
                            p.force_scale,
                            p.particle_size,
                        ]
                        .iter()
                        .all(|x| x.is_finite() && *x >= 0.0)
                })
                .unwrap_or(true)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_creature_loads_without_velocities() {
        let p:Prefab=serde_json::from_str(r#"{"name":"old","particles":[{"relative_position":[0,0,0],"kind":0}],"force_matrix":[1],"type_count":1,"particle_count":1,"notes":""}"#).unwrap();
        assert!(p.valid());
        assert!(p.particles[0].velocity.is_empty());
        assert_eq!(p.dimension, 3);
    }
    #[test]
    fn failed_save_retains_dirty_state() {
        let mut b = Book::new();
        b.path = "/dev/null/book.json".into();
        b.mark_dirty();
        assert!(b.flush_if_dirty().unwrap().is_err());
        assert!(b.dirty);
    }
}
