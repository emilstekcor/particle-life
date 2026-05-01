use serde::Serialize;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{sim::SimState, ui::UiState};

#[derive(Serialize)]
pub struct CrashProfile {
    timestamp_unix: u64,

    particle_count: usize,
    step_count: u64,

    use_gpu_physics: bool,
    paused: bool,

    type_count: usize,
    bounds: f32,
    dt: f32,
    r_max: f32,
    force_scale: f32,
    friction: f32,
    beta: f32,
    max_speed: f32,
    wrap: bool,

    reactions_enabled: bool,
    mix_radius: f32,
    reaction_probability: f32,
    preserve_particle_count: bool,

    trace_len: u32,
    trace_type_filter: i32,

    last_step_used_grid: bool,
    last_neighbor_checks: u64,
    last_grid_res: usize,
    last_step_ms: f32,
    avg_step_ms: f32,

    force_matrix: Vec<f32>,
    reaction_table: Vec<i32>,
}

pub fn crash_profile_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("last_crash_profile.json")
}

pub fn save_crash_profile(sim: &SimState, ui: &UiState) {
    let path = crash_profile_path();

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let profile = CrashProfile {
        timestamp_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),

        particle_count: sim.particles.len(),
        step_count: sim.step_count,

        use_gpu_physics: ui.use_gpu_physics,
        paused: ui.paused,

        type_count: sim.params.type_count,
        bounds: sim.params.bounds,
        dt: sim.params.dt,
        r_max: sim.params.r_max,
        force_scale: sim.params.force_scale,
        friction: sim.params.friction,
        beta: sim.params.beta,
        max_speed: sim.params.max_speed,
        wrap: sim.params.wrap,

        reactions_enabled: sim.params.reactions_enabled,
        mix_radius: sim.params.mix_radius,
        reaction_probability: sim.params.reaction_probability,
        preserve_particle_count: sim.params.preserve_particle_count,

        trace_len: ui.trace_len,
        trace_type_filter: ui.trace_type_filter,

        last_step_used_grid: sim.last_step_used_grid,
        last_neighbor_checks: sim.last_neighbor_checks,
        last_grid_res: sim.last_grid_res,
        last_step_ms: sim.last_step_ms,
        avg_step_ms: sim.avg_step_ms,

        force_matrix: sim.force_matrix.clone(),
        reaction_table: sim.reaction_table.clone(),
    };

    if let Ok(json) = serde_json::to_string_pretty(&profile) {
        let _ = fs::write(path, json);
    }
}
