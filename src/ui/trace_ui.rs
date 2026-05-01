use egui::*;
use crate::sim::SimState;
use crate::ui::UiState;

pub fn draw_trace_controls(ui: &mut Ui, sim: &mut SimState, state: &mut UiState) {
    ui.horizontal(|ui| {
        ui.checkbox(&mut state.traces, "Enable traces");
        ui.checkbox(&mut state.trace_ui_edit_only, "Edit matrix only");

        egui::ComboBox::from_label("Render")
            .selected_text(format!("{:?}", state.trace_render_mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Off, "Off");
                ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Simple, "Simple");
                ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Lines, "Lines");
                ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Dots, "Dots");
            });
    });

    // Advanced trail modes
    ui.label("Advanced Trails:");

    // Trail mode dropdown
    egui::ComboBox::from_label("Trail Mode")
        .selected_text(match state.trace_render_mode {
            crate::ui::TraceRenderMode::Off => "Off",
            crate::ui::TraceRenderMode::Simple => "Simple",
            crate::ui::TraceRenderMode::Lines => "Lines",
            crate::ui::TraceRenderMode::Dots => "Dots",
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Off, "Off");
            ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Simple, "Simple");
            ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Lines, "Lines");
            ui.selectable_value(&mut state.trace_render_mode, crate::ui::TraceRenderMode::Dots, "Dots");
        });

    // Trail type filter dropdown
    egui::ComboBox::from_label("Trail Type")
        .selected_text(if state.trace_type_filter < 0 {
            "All".to_string()
        } else {
            format!("Type {}", state.trace_type_filter)
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut state.trace_type_filter, -1, "All");
            for i in 0..sim.params.type_count {
                ui.selectable_value(&mut state.trace_type_filter, i as i32, format!("Type {}", i));
            }
        });

    ui.add(egui::Slider::new(&mut state.trace_len, 1..=16).text("Trail Length"));
    ui.checkbox(&mut state.trace_trigger_only, "Only capture reacting particles");
    #[cfg(debug_assertions)]
    ui.checkbox(&mut state.debug_trails, "Trail Debug Output");
    ui.separator();
}

pub fn draw_trace_matrix_ui(ui: &mut Ui, sim: &mut SimState, state: &mut UiState) {
    let n = sim.params.type_count;
    if n == 0 {
        return;
    }

    ui.horizontal(|ui| {
        ui.label("Trace Length Matrix:");
        ui.add_space(20.0);
        
        if ui.button("Randomize").clicked() {
            sim.randomize_rules();
        }
        if ui.button("Zero").clicked() {
            // Zero out force matrix
            for i in 0..sim.params.type_count {
                for j in 0..sim.params.type_count {
                    sim.set_rule(i, j, 0.0);
                }
            }
        }
        if ui.button("Clear").clicked() {
            // Clear force matrix (same as zero)
            for i in 0..sim.params.type_count {
                for j in 0..sim.params.type_count {
                    sim.set_rule(i, j, 0.0);
                }
            }
        }
    });

    // Trace length matrix editor
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
                        .speed(1)
                        .clamp_range(0.0..=16.0)
                        .suffix(" steps")
                );
                
                if response.changed() {
                    sim.set_rule(i, j, value);
                }
                
                // Color code based on value
                if value > 8.0 {
                    ui.label(egui::RichText::new("●").color(egui::Color32::GREEN));
                } else if value > 0.0 {
                    ui.label(egui::RichText::new("●").color(egui::Color32::YELLOW));
                } else {
                    ui.label("○");
                }
            }
        });
    }

    // Copy/Paste and Presets for trace matrix
    ui.horizontal(|ui| {
        if ui.button("Copy").clicked() {
            let mut matrix = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    matrix.push(sim.get_rule(i, j) as u32);
                }
            }
            state.trace_clipboard = Some(matrix);
        }
        if ui.button("Paste").clicked() {
            if let Some(ref matrix) = state.trace_clipboard {
                if matrix.len() == n * n {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_rule(i, j, matrix[i * n + j] as f32);
                        }
                    }
                }
            }
        }
        
        egui::ComboBox::from_label("Presets")
            .selected_text("Select preset")
            .show_ui(ui, |ui| {
                if ui.selectable_value(&mut (), (), "Short Trails").clicked() {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_rule(i, j, 4.0);
                        }
                    }
                }
                if ui.selectable_value(&mut (), (), "Long Trails").clicked() {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_rule(i, j, 16.0);
                        }
                    }
                }
                if ui.selectable_value(&mut (), (), "No Trails").clicked() {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_rule(i, j, 0.0);
                        }
                    }
                }
            });
    });
}

pub fn draw_trace_debug(ui: &mut Ui, _sim: &mut SimState, state: &mut UiState) {
    ui.collapsing("Trace Debug", |ui| {
        ui.label(format!("Traces Enabled: {}", state.traces));
        ui.label(format!("Trace Render Mode: {:?}", state.trace_render_mode));
        ui.label(format!("Trace Length: {}", state.trace_len));
        ui.label(format!("Trace Type Filter: {}", state.trace_type_filter));
        ui.label(format!("Trace Trigger Only: {}", state.trace_trigger_only));
        ui.label(format!("Trace UI Edit Only: {}", state.trace_ui_edit_only));
        ui.label(format!("Debug Trails: {}", state.debug_trails));
        
        #[cfg(debug_assertions)]
        ui.checkbox(&mut state.debug_trails, "Trail Debug Output");
    });
}
