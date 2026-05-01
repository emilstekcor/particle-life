use egui::*;
use crate::sim::SimState;
use crate::ui::{UiState, SelectionMode};

pub fn draw_selection_controls(ui: &mut Ui, sim: &mut SimState, state: &mut UiState) {
    ui.horizontal(|ui| {
        ui.label("Selection Mode:");
        ui.radio_value(&mut state.selection_mode, SelectionMode::Rect, "Rect");
        ui.radio_value(&mut state.selection_mode, SelectionMode::Brush, "Brush");
        ui.radio_value(&mut state.selection_mode, SelectionMode::Slice, "Slice");
    });

    match state.selection_mode {
        SelectionMode::Brush => {
            ui.add(egui::Slider::new(&mut state.brush_radius, 5.0..=100.0).text("Brush Radius"));
        }
        SelectionMode::Slice => {
            ui.add(egui::Slider::new(&mut state.slice_center, -0.5..=1.5).text("Slice Center"));
            ui.add(egui::Slider::new(&mut state.slice_thickness, 0.01..=0.5).text("Slice Thickness"));
        }
        _ => {}
    }

    ui.separator();

    // Selection info and actions
    ui.horizontal(|ui| {
        ui.label(format!("Selected: {} particles", state.selected_indices.len()));
        
        if ui.button("Clear Selection").clicked() {
            state.selected_indices.clear();
            state.highlighted_indices.clear();
        }
    });

    // Selection-based operations
    if !state.selected_indices.is_empty() {
        ui.separator();
        ui.label("Selection Operations:");
        
        ui.horizontal(|ui| {
            if ui.button("Delete Selected").clicked() {
                sim.delete_particles(&state.selected_indices.clone());
                state.selected_indices.clear();
                state.highlighted_indices.clear();
            }
            
            if ui.button("Duplicate Selected").clicked() {
                sim.duplicate_particles(&state.selected_indices);
            }
        });
        
        ui.horizontal(|ui| {
            if ui.button("Cool Down").clicked() {
                sim.scale_velocities(0.1);
            }
            
            if ui.button("Energy Boost").clicked() {
                sim.scale_velocities(2.0);
            }
        });
    }

    // Move controls
    if !state.selected_indices.is_empty() {
        ui.separator();
        ui.label("Move Selection:");
        
        ui.horizontal(|ui| {
            if ui.button("Move Up").clicked() {
                sim.move_particles(&state.selected_indices, glam::Vec3::new(0.0, 0.05, 0.0));
            }
            if ui.button("Move Down").clicked() {
                sim.move_particles(&state.selected_indices, glam::Vec3::new(0.0, -0.05, 0.0));
            }
            if ui.button("Move Left").clicked() {
                sim.move_particles(&state.selected_indices, glam::Vec3::new(-0.05, 0.0, 0.0));
            }
            if ui.button("Move Right").clicked() {
                sim.move_particles(&state.selected_indices, glam::Vec3::new(0.05, 0.0, 0.0));
            }
        });
        
        ui.horizontal(|ui| {
            if ui.button("Move Forward").clicked() {
                sim.move_particles(&state.selected_indices, glam::Vec3::new(0.0, 0.0, 0.05));
            }
            if ui.button("Move Back").clicked() {
                sim.move_particles(&state.selected_indices, glam::Vec3::new(0.0, 0.0, -0.05));
            }
        });
    }
}

pub fn draw_selection_debug(ui: &mut Ui, _sim: &mut SimState, state: &mut UiState) {
    ui.collapsing("Selection Debug", |ui| {
        ui.label(format!("Mode: {:?}", state.selection_mode));
        ui.label(format!("Drag Mode: {:?}", state.drag_mode));
        ui.label(format!("Is Selecting: {}", state.is_selecting));
        ui.label(format!("Selection Readback Needed: {}", state.selection_readback_needed));
        
        if let Some(start) = state.drag_start {
            ui.label(format!("Drag Start: ({:.2}, {:.2})", start.x, start.y));
        }
        if let Some(end) = state.drag_end {
            ui.label(format!("Drag End: ({:.2}, {:.2})", end.x, end.y));
        }
        
        ui.label(format!("Selected Count: {}", state.selected_indices.len()));
        ui.label(format!("Highlighted Count: {}", state.highlighted_indices.len()));
        
        if !state.selected_indices.is_empty() {
            ui.label("Selected indices:");
            ui.label(format!("{:?}", &state.selected_indices[..state.selected_indices.len().min(10)]));
            if state.selected_indices.len() > 10 {
                ui.label("...");
            }
        }
    });
}
