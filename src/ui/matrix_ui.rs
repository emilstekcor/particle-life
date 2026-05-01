use egui::*;
use crate::sim::SimState;
use crate::ui::UiState;

pub fn draw_force_matrix_ui(ui: &mut Ui, sim: &mut SimState, _state: &mut UiState) {
    let n = sim.params.type_count;
    if n == 0 {
        return;
    }

    ui.horizontal(|ui| {
        ui.label("Force Matrix:");
        ui.add_space(20.0);
        
        if ui.button("Randomize").clicked() {
            sim.randomize_rules();
        }
        if ui.button("Zero").clicked() {
            for i in 0..sim.params.type_count {
                for j in 0..sim.params.type_count {
                    sim.set_rule(i, j, 0.0);
                }
            }
        }
        if ui.button("Clear").clicked() {
            for i in 0..sim.params.type_count {
                for j in 0..sim.params.type_count {
                    sim.set_rule(i, j, 0.0);
                }
            }
        }
        if ui.button("Raise All").clicked() {
            for i in 0..sim.params.type_count {
                for j in 0..sim.params.type_count {
                    let current = sim.get_rule(i, j);
                    let raised = (current + 0.1).min(1.0).max(-1.0);
                    sim.set_rule(i, j, raised);
                }
            }
        }
        if ui.button("Lower All").clicked() {
            for i in 0..sim.params.type_count {
                for j in 0..sim.params.type_count {
                    let current = sim.get_rule(i, j);
                    let lowered = (current - 0.1).min(1.0).max(-1.0);
                    sim.set_rule(i, j, lowered);
                }
            }
        }
    });

    // Force matrix editor
    ui.horizontal(|ui| {
        // Column headers
        ui.add_space(20.0);
        for j in 0..n {
            ui.label(format!("T{}", j));
        }
    });

    for i in 0..n {
        ui.horizontal(|ui| {
            // Row header
            ui.label(format!("T{}", i));
            
            // Matrix cells
            for j in 0..n {
                let mut value = sim.get_rule(i, j);
                let response = ui.add(
                    egui::DragValue::new(&mut value)
                        .speed(0.01)
                        .clamp_range(-1.0..=1.0)
                        .fixed_decimals(2)
                );
                
                if response.changed() {
                    sim.set_rule(i, j, value);
                }
                
                // Color code based on value
                if value > 0.0 {
                    ui.label(egui::RichText::new("●").color(egui::Color32::GREEN));
                } else if value < 0.0 {
                    ui.label(egui::RichText::new("●").color(egui::Color32::RED));
                } else {
                    ui.label("○");
                }
            }
        });
    }
}

pub fn draw_reaction_matrix_ui(ui: &mut Ui, sim: &mut SimState, state: &mut UiState) {
    let n = sim.params.type_count;
    if n == 0 {
        return;
    }

    ui.horizontal(|ui| {
        ui.label("Reaction Matrix:");
        ui.add_space(20.0);
        
        if ui.button("Randomize").clicked() {
            sim.default_reaction_table();
        }
        if ui.button("Zero").clicked() {
            sim.resize_reaction_table();
        }
        if ui.button("Clear").clicked() {
            sim.resize_reaction_table();
        }
    });

    // Reaction matrix editor
    ui.horizontal(|ui| {
        // Column headers
        ui.add_space(20.0);
        for j in 0..n {
            ui.label(format!("T{}", j));
        }
    });

    for i in 0..n {
        ui.horizontal(|ui| {
            // Row header
            ui.label(format!("T{}", i));
            
            // Matrix cells
            for j in 0..n {
                let mut value = sim.rx(i, j) as f32;
                let response = ui.add(
                    egui::DragValue::new(&mut value)
                        .speed(0.01)
                        .clamp_range(-1.0..=1.0)
                        .fixed_decimals(2)
                );
                
                if response.changed() {
                    sim.set_reaction(i, j, value as i32);
                }
                
                // Color code based on value
                if value > 0.0 {
                    ui.label(egui::RichText::new("●").color(egui::Color32::GREEN));
                } else if value < 0.0 {
                    ui.label(egui::RichText::new("●").color(egui::Color32::RED));
                } else {
                    ui.label("○");
                }
            }
        });
    }

    // Safety guard warning for GPU reactions
    if state.use_gpu_physics && sim.params.reactions_enabled && sim.particles.len() > 10_000 {
        ui.separator();
        ui.colored_label(
            egui::Color32::RED,
            "⚠️ GPU reactions disabled above 10k particles until grid reaction pass exists."
        );
        ui.colored_label(
            egui::Color32::YELLOW,
            format!("Current: {} particles (threshold: 10,000)", sim.particles.len())
        );
        ui.separator();
    }

    // Copy/Paste and Presets for reaction matrix
    ui.horizontal(|ui| {
        if ui.button("Copy").clicked() {
            let mut matrix = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    matrix.push(sim.rx(i, j));
                }
            }
            state.reaction_clipboard = Some(matrix);
        }
        if ui.button("Paste").clicked() {
            if let Some(ref matrix) = state.reaction_clipboard {
                if matrix.len() == n * n {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_reaction(i, j, matrix[i * n + j]);
                        }
                    }
                }
            }
        }
        
        egui::ComboBox::from_label("Presets")
            .selected_text("Select preset")
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut (), (), "Random");
                ui.selectable_value(&mut (), (), "Predator-Prey");
                ui.selectable_value(&mut (), (), "Symbiotic");
                ui.selectable_value(&mut (), (), "Competitive");
            });
    });
}
