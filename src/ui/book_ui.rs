use egui::*;
use crate::sim::SimState;
use crate::ui::UiState;

pub fn draw_book_ui(ui: &mut Ui, sim: &mut SimState, state: &mut UiState) {
    ui.heading("Creature Book");
    ui.separator();

    // Book controls
    ui.horizontal(|ui| {
        if ui.button("Save Current").clicked() {
            // TODO: Add save current creature to book
        }
        if ui.button("New Entry").clicked() {
            // TODO: Add new creature entry
        }
        if ui.button("Delete Selected").clicked() {
            // TODO: Delete selected creature from book
        }
    });

    ui.separator();

    // Creature list
    ui.collapsing("Creature Library", |ui| {
        let book = &sim.book;
        
        if book.prefabs.is_empty() {
            ui.label("No prefabs saved yet");
            return;
        }

        for (i, prefab) in book.prefabs.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.selectable_label(state.selected_creature == Some(i), &prefab.name).clicked() {
                    state.selected_creature = Some(i);
                }
                
                if ui.button("Load").clicked() {
                    // TODO: Load prefab from book
                    // sim.load_from_book(&prefab.name);
                }
                
                if ui.button("Delete").clicked() {
                    // TODO: Delete prefab from book
                }
            });
        }
    });

    // Selected creature details
    if let Some(selected_idx) = state.selected_creature {
        if let Some(prefab) = sim.book.prefabs.get(selected_idx) {
            ui.separator();
            ui.heading(format!("Prefab: {}", prefab.name));
            
            // TODO: Add prefab details when structure is known
            ui.label("Prefab details not yet implemented");
            
            // Load button
            if ui.button("Load This Prefab").clicked() {
                // TODO: Implement prefab loading
                // sim.load_from_book(&prefab.name);
            }
        }
    }

    // Import/Export
    ui.separator();
    ui.heading("Import/Export");
    
    ui.horizontal(|ui| {
        if ui.button("Export Book").clicked() {
            // TODO: Export book to file
        }
        if ui.button("Import Book").clicked() {
            // TODO: Import book from file
        }
    });
}

pub fn draw_profile_ui(ui: &mut Ui, _sim: &mut SimState, state: &mut UiState) {
    ui.heading("Crash Profiles");
    ui.separator();

    // Profile controls
    ui.horizontal(|ui| {
        if ui.button("Save Profile").clicked() {
            state.save_profile_now = true;
        }
        if ui.button("Load Profile").clicked() {
            // TODO: Implement profile loading
        }
        if ui.button("Clear Profiles").clicked() {
            // TODO: Clear all profiles
        }
    });

    ui.separator();

    // Profile list
    ui.collapsing("Saved Profiles", |ui| {
        // TODO: List available crash profiles
        ui.label("Profile listing not yet implemented");
    });

    // Auto-save settings
    ui.collapsing("Auto-Save Settings", |ui| {
        ui.checkbox(&mut state.auto_save_profiles, "Enable auto-save");
        
        if state.auto_save_profiles {
            ui.add(egui::Slider::new(&mut state.auto_save_interval, 100..=5000).text("Save every N steps"));
        }
    });
}
