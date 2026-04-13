use serde::{Deserialize, Serialize};
use std::fs;

// ── A single particle's data within a prefab ──────────────────────────────────
// Positions are relative to the prefab's center of mass
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefabParticle {
    pub relative_position: [f32; 3],
    pub kind: u32,
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

    // Metadata
    pub particle_count: usize,
    pub notes: String,
}

impl Prefab {
    pub fn new(name: String, force_matrix: Vec<f32>, type_count: usize) -> Self {
        Self {
            name,
            particles: Vec::new(),
            force_matrix,
            type_count,
            particle_count: 0,
            notes: String::new(),
        }
    }
}

// ── The Book — the full creature library ─────────────────────────────────────
#[derive(Debug, Default)]
pub struct Book {
    pub prefabs: Vec<Prefab>,
    pub path: String,
}

impl Book {
    pub fn new() -> Self {
        Self {
            prefabs: Vec::new(),
            path: "book.json".to_string(),
        }
    }

    /// Load prefabs from book.json. Silent no-op if file doesn't exist yet.
    pub fn load_from_file(&mut self, path: &str) {
        self.path = path.to_string();
        match fs::read_to_string(path) {
            Ok(contents) => {
                match serde_json::from_str::<Vec<Prefab>>(&contents) {
                    Ok(prefabs) => {
                        self.prefabs = prefabs;
                        log::info!("Loaded {} creature(s) from {}", self.prefabs.len(), path);
                    }
                    Err(e) => log::warn!("Failed to parse {}: {}", path, e),
                }
            }
            Err(_) => {
                // File doesn't exist yet — that's fine, we'll create it on first save
                log::info!("No {} found, starting with empty book", path);
            }
        }
    }

    /// Save all prefabs to book.json
    pub fn save_to_file(&self) {
        match serde_json::to_string_pretty(&self.prefabs) {
            Ok(json) => {
                if let Err(e) = fs::write(&self.path, json) {
                    log::error!("Failed to write {}: {}", self.path, e);
                } else {
                    log::info!("Saved {} creature(s) to {}", self.prefabs.len(), self.path);
                }
            }
            Err(e) => log::error!("Failed to serialize book: {}", e),
        }
    }

    pub fn add_prefab(&mut self, prefab: Prefab) {
        self.prefabs.push(prefab);
        self.save_to_file();
    }

    pub fn remove_prefab(&mut self, index: usize) {
        if index < self.prefabs.len() {
            self.prefabs.remove(index);
            self.save_to_file();
        }
    }
}
