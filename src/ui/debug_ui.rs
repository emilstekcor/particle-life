use egui::*;
use crate::sim::SimState;
use crate::ui::UiState;

pub fn draw_debug_panel(ui: &mut Ui, sim: &mut SimState, state: &mut UiState) {
    ui.heading("Debug Panel");
    ui.separator();

    // Simulation debug info
    ui.collapsing("Simulation", |ui| {
        ui.label(format!("Particles: {}", sim.particles.len()));
        ui.label(format!("Step Count: {}", sim.step_count));
        ui.label(format!("Type Count: {}", sim.params.type_count));
        ui.label(format!("Bounds: {:.2}", sim.params.bounds));
        ui.label(format!("Time Step: {:.4}", sim.params.dt));
        ui.label(format!("Max Speed: {:.2}", sim.params.max_speed));
        ui.label(format!("Beta: {:.2}", sim.params.beta));
        ui.label(format!("Reactions Enabled: {}", sim.params.reactions_enabled));
        ui.label(format!("Mix Radius: {:.3}", sim.params.mix_radius));
        ui.label(format!("Reaction Probability: {:.3}", sim.params.reaction_probability));
    });

    // Physics debug info
    ui.collapsing("Physics", |ui| {
        ui.label(format!("GPU Physics: {}", state.use_gpu_physics));
        ui.label(format!("Paused: {}", state.paused));
        ui.label(format!("Step Once: {}", state.step_once));
        ui.label(format!("Cap to Bounds: {}", state.cap_to_bounds));
    });

    // Camera debug info
    ui.collapsing("Camera", |ui| {
        ui.label(format!("Camera Mode: {}", state.camera_mode));
        ui.label(format!("Position: ({:.2}, {:.2}, {:.2})", state.fly_pos.x, state.fly_pos.y, state.fly_pos.z));
        ui.label(format!("Yaw: {:.3} rad", state.fly_yaw));
        ui.label(format!("Pitch: {:.3} rad", state.fly_pitch));
        ui.label(format!("Speed: {:.2}", state.fly_speed));
        ui.label(format!("Mouse Look Active: {}", state.mouse_look_active));
    });

    // Selection debug info
    ui.collapsing("Selection", |ui| {
        ui.label(format!("Selection Mode: {:?}", state.selection_mode));
        ui.label(format!("Drag Mode: {:?}", state.drag_mode));
        ui.label(format!("Is Selecting: {}", state.is_selecting));
        ui.label(format!("Selection Readback Needed: {}", state.selection_readback_needed));
        ui.label(format!("Selected Count: {}", state.selected_indices.len()));
        ui.label(format!("Highlighted Count: {}", state.highlighted_indices.len()));
        ui.label(format!("Brush Radius: {:.1}", state.brush_radius));
        
        if let Some(start) = state.drag_start {
            ui.label(format!("Drag Start: ({:.1}, {:.1})", start.x, start.y));
        }
        if let Some(end) = state.drag_end {
            ui.label(format!("Drag End: ({:.1}, {:.1})", end.x, end.y));
        }
    });

    // Performance debug info
    ui.collapsing("Performance", |ui| {
        ui.label(format!("Viewport: {}x{}", state.viewport[0], state.viewport[1]));
        ui.label(format!("Slice Center: {:.3}", state.slice_center));
        ui.label(format!("Slice Thickness: {:.3}", state.slice_thickness));
        
        if let Some(hover) = state.hover_index {
            ui.label(format!("Hover Index: {}", hover));
        }
    });

    // Memory usage
    ui.collapsing("Memory", |ui| {
        let particle_size = std::mem::size_of::<crate::sim::Particle>();
        let gpu_particle_size = std::mem::size_of::<crate::sim::GpuParticle>();
        let selection_params_size = std::mem::size_of::<crate::renderer::compute::SelectionParams>();
        let trail_point_size = std::mem::size_of::<crate::renderer::compute::TrailPoint>();
        
        ui.label(format!("Particle (CPU): {} bytes", particle_size));
        ui.label(format!("Particle (GPU): {} bytes", gpu_particle_size));
        ui.label(format!("SelectionParams: {} bytes", selection_params_size));
        ui.label(format!("TrailPoint: {} bytes", trail_point_size));
        
        let total_particle_memory = sim.particles.len() * particle_size;
        ui.label(format!("Total Particle Memory: {} KB", total_particle_memory / 1024));
    });

    // Force matrix debug
    ui.collapsing("Force Matrix", |ui| {
        let n = sim.params.type_count;
        ui.label(format!("Size: {}x{}", n, n));
        
        if n > 0 {
            ui.horizontal(|ui| {
                ui.label("Sample values:");
                for i in 0..n.min(3) {
                    for j in 0..n.min(3) {
                        ui.label(format!("{:.2}", sim.rx(i, j)));
                    }
                    if n > 3 {
                        ui.label("...");
                    }
                }
                if n > 3 {
                    ui.label("...");
                }
            });
        }
    });

    // Reaction matrix debug
    ui.collapsing("Reaction Matrix", |ui| {
        let n = sim.params.type_count;
        ui.label(format!("Size: {}x{}", n, n));
        
        if n > 0 {
            ui.horizontal(|ui| {
                ui.label("Sample values:");
                for i in 0..n.min(3) {
                    for j in 0..n.min(3) {
                        ui.label(format!("{:.2}", sim.rx(i, j)));
                    }
                    if n > 3 {
                        ui.label("...");
                    }
                }
                if n > 3 {
                    ui.label("...");
                }
            });
        }
    });

    // Crash profile controls
    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Save Crash Profile").clicked() {
            state.save_profile_now = true;
        }
        if ui.button("Load Crash Profile").clicked() {
            // TODO: Implement crash profile loading
        }
    });
}
