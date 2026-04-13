use glam::Vec3;
use crate::sim::{Particle, SimParams};

/// World boundary and utility functions
pub struct World {
    pub bounds: f32,
    pub wrap: bool,
}

impl World {
    pub fn new(params: &SimParams) -> Self {
        Self {
            bounds: params.bounds,
            wrap: params.wrap,
        }
    }

    /// Apply boundary conditions to a position
    pub fn apply_boundaries(&self, pos: &mut Vec3) {
        if self.wrap {
            // Toroidal wrapping
            if pos.x < 0.0 { pos.x += self.bounds; }
            else if pos.x >= self.bounds { pos.x -= self.bounds; }
            
            if pos.y < 0.0 { pos.y += self.bounds; }
            else if pos.y >= self.bounds { pos.y -= self.bounds; }
            
            if pos.z < 0.0 { pos.z += self.bounds; }
            else if pos.z >= self.bounds { pos.z -= self.bounds; }
        } else {
            // Hard boundaries - clamp to world bounds
            pos.x = pos.x.clamp(0.0, self.bounds);
            pos.y = pos.y.clamp(0.0, self.bounds);
            pos.z = pos.z.clamp(0.0, self.bounds);
        }
    }

    /// Calculate wrapped distance between two points (for toroidal topology)
    pub fn wrapped_distance(&self, a: Vec3, b: Vec3) -> Vec3 {
        if !self.wrap {
            return b - a;
        }

        let mut diff = b - a;
        let half = self.bounds * 0.5;

        // Wrap each component to the shortest distance
        if diff.x > half { diff.x -= self.bounds; }
        else if diff.x < -half { diff.x += self.bounds; }

        if diff.y > half { diff.y -= self.bounds; }
        else if diff.y < -half { diff.y += self.bounds; }

        if diff.z > half { diff.z -= self.bounds; }
        else if diff.z < -half { diff.z += self.bounds; }

        diff
    }

    /// Check if a position is within world bounds
    pub fn is_in_bounds(&self, pos: Vec3) -> bool {
        pos.x >= 0.0 && pos.x < self.bounds &&
        pos.y >= 0.0 && pos.y < self.bounds &&
        pos.z >= 0.0 && pos.z < self.bounds
    }

    /// Get a random position within the world
    pub fn random_position(&self) -> Vec3 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Vec3::new(
            rng.gen_range(0.0..self.bounds),
            rng.gen_range(0.0..self.bounds),
            rng.gen_range(0.0..self.bounds),
        )
    }

    /// Get the center of the world
    pub fn center(&self) -> Vec3 {
        Vec3::splat(self.bounds * 0.5)
    }

    /// Get the world size as a Vec3
    pub fn size(&self) -> Vec3 {
        Vec3::splat(self.bounds)
    }
}