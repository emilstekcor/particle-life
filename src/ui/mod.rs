use egui::{Context, Slider, Grid, RichText, Color32};
use crate::sim::SimState;

// ── UiState ───────────────────────────────────────────────────────────────────
// All the UI-owned state that doesn't live in SimState
pub struct UiState {
    pub paused:       bool,
    pub step_once:    bool,

    // Orbit camera
    pub camera_yaw:   f32,
    pub camera_pitch: f32,
    pub camera_dist:  f32,

    // Book panel
    pub show_book:    bool,
    pub new_prefab_name: String,

    // Selection
    pub selected_prefab: Option<usize>,  // index into sim.book.prefabs
}

impl UiState {
    pub fn new() -> Self {
        Self {
            paused: false,
            step_once: false,
            camera_yaw: 0.5,
            camera_pitch: 0.3,
            camera_dist: 1.5,
            show_book: false,
            new_prefab_name: String::new(),
            selected_prefab: None,
        }
    }
}

// ── Main UI draw function — called from Renderer::render each frame ───────────
pub fn draw_ui(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    draw_controls(ctx, sim, ui);
    draw_rule_matrix(ctx, sim);
    if ui.show_book {
        draw_book(ctx, sim, ui);
    }
}

// ── Controls panel ────────────────────────────────────────────────────────────
fn draw_controls(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    egui::Window::new("Controls")
        .default_pos([12.0, 12.0])
        .resizable(false)
        .show(ctx, |e| {
            // Status
            e.label(format!("Particles: {}", sim.particles.len()));
            e.label(format!("Types: {}", sim.params.type_count));
            e.label(format!("Steps: {}", sim.step_count));
            e.separator();

            // Sim controls
            e.horizontal(|e| {
                let pause_label = if ui.paused { "▶ Resume" } else { "⏸ Pause" };
                if e.button(pause_label).clicked() {
                    ui.paused = !ui.paused;
                }
                if e.button("⏭ Step").clicked() {
                    ui.step_once = true;
                }
            });
            e.separator();

            // Params
            e.label("Parameters");
            let mut changed = false;
            changed |= e.add(Slider::new(&mut sim.params.r_max, 0.01..=0.5).text("r_max")).changed();
            changed |= e.add(Slider::new(&mut sim.params.force_scale, 0.1..=5.0).text("force scale")).changed();
            changed |= e.add(Slider::new(&mut sim.params.friction, 0.01..=1.0).text("friction")).changed();
            changed |= e.add(Slider::new(&mut sim.params.dt, 0.001..=0.05).text("dt")).changed();
            changed |= e.checkbox(&mut sim.params.wrap, "Wrap").changed();
            if changed {
                sim.params_dirty = true;
            }
            e.separator();

            // Particle count
            e.label("Spawn");
            e.horizontal(|e| {
                if e.button("-100").clicked() {
                    let new = sim.particles.len().saturating_sub(100);
                    sim.particles.truncate(new);
                    sim.particles_dirty = true;
                }
                if e.button("+100").clicked() {
                    sim.spawn_random(100);
                }
                if e.button("Clear").clicked() {
                    sim.clear_particles();
                }
            });
            e.separator();

            // Rules
            if e.button("🎲 Randomize Rules").clicked() {
                sim.randomize_rules();
            }
            e.separator();

            // Camera
            e.label("Camera");
            e.add(Slider::new(&mut ui.camera_dist, 0.1..=5.0).text("distance"));
            e.add(Slider::new(&mut ui.camera_yaw, -3.14..=3.14).text("yaw"));
            e.add(Slider::new(&mut ui.camera_pitch, -1.5..=1.5).text("pitch"));
            e.separator();

            // Book toggle
            if e.button("📖 Creature Book").clicked() {
                ui.show_book = !ui.show_book;
            }
        });
}

// ── Rule matrix panel ─────────────────────────────────────────────────────────
fn draw_rule_matrix(ctx: &Context, sim: &mut SimState) {
    let n = sim.params.type_count;

    egui::Window::new("Rule Matrix")
        .default_pos([350.0, 12.0])
        .resizable(false)
        .show(ctx, |e| {
            e.label("Left-click cell to increase, right-click to decrease");
            e.separator();

            Grid::new("matrix").spacing([2.0, 2.0]).show(e, |e| {
                // Header row
                e.label("");
                for col in 0..n {
                    e.label(RichText::new(format!("T{}", col))
                        .color(type_color_egui(col)));
                }
                e.end_row();

                for row in 0..n {
                    e.label(RichText::new(format!("T{}", row))
                        .color(type_color_egui(row)));

                    for col in 0..n {
                        let val = sim.get_rule(row, col);

                        // Color: green = attract, red = repel, grey = neutral
                        let cell_color = if val > 0.0 {
                            Color32::from_rgb(
                                30, (40.0 + val * 180.0) as u8, 30
                            )
                        } else if val < 0.0 {
                            Color32::from_rgb(
                                (40.0 + val.abs() * 180.0) as u8, 30, 30
                            )
                        } else {
                            Color32::from_rgb(40, 40, 40)
                        };

                        let label = format!("{:.1}", val);
                        let btn = egui::Button::new(
                            RichText::new(label).size(12.0)
                        )
                        .fill(cell_color)
                        .min_size(egui::vec2(38.0, 28.0));

                        let resp = e.add(btn);

                        if resp.clicked() {
                            sim.set_rule(row, col, (val + 0.1).clamp(-1.0, 1.0));
                        }
                        if resp.secondary_clicked() {
                            sim.set_rule(row, col, (val - 0.1).clamp(-1.0, 1.0));
                        }
                        // Scroll wheel on a cell
                        if resp.hovered() {
                            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                            if scroll != 0.0 {
                                sim.set_rule(row, col, (val + scroll * 0.01).clamp(-1.0, 1.0));
                            }
                        }
                    }
                    e.end_row();
                }
            });
        });
}

// ── Creature Book panel ───────────────────────────────────────────────────────
fn draw_book(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    egui::Window::new("📖 Creature Book")
        .default_pos([12.0, 500.0])
        .default_width(300.0)
        .show(ctx, |e| {
            e.label(format!("{} creature(s) saved", sim.book.prefabs.len()));
            e.separator();

            // Save current selection as a new prefab
            e.label("Save new creature:");
            e.horizontal(|e| {
                e.text_edit_singleline(&mut ui.new_prefab_name);
                if e.button("💾 Save").clicked() && !ui.new_prefab_name.is_empty() {
                    save_selection_as_prefab(sim, ui.new_prefab_name.clone());
                    ui.new_prefab_name.clear();
                }
            });
            e.separator();

            // List saved creatures
            let mut to_remove: Option<usize> = None;
            for (i, prefab) in sim.book.prefabs.iter().enumerate() {
                e.horizontal(|e| {
                    let selected = ui.selected_prefab == Some(i);
                    if e.selectable_label(selected, &prefab.name).clicked() {
                        ui.selected_prefab = Some(i);
                    }
                    e.label(format!("({} particles)", prefab.particle_count));
                    if e.button("🗑").clicked() {
                        to_remove = Some(i);
                    }
                });
            }

            if let Some(i) = to_remove {
                sim.book.remove_prefab(i);
                if ui.selected_prefab == Some(i) {
                    ui.selected_prefab = None;
                }
            }

            if let Some(i) = ui.selected_prefab {
                if i < sim.book.prefabs.len() {
                    e.separator();
                    if e.button("▶ Spawn selected creature").clicked() {
                        use glam::Vec3;
                        let center = Vec3::splat(sim.params.bounds * 0.5);
                        sim.spawn_prefab(i, center, 0);
                    }
                }
            }
        });
}

/// Capture particles with prefab_id >= 0 and save them to the book.
/// For now saves ALL tagged particles — later this will be selection-based.
fn save_selection_as_prefab(sim: &mut SimState, name: String) {
    use crate::sim::book::{Prefab, PrefabParticle};
    use glam::Vec3;

    // Collect tagged particles
    let tagged: Vec<_> = sim.particles.iter()
        .filter(|p| p.prefab_id >= 0)
        .collect();

    if tagged.is_empty() {
        log::warn!("No tagged particles to save as creature");
        return;
    }

    // Compute center of mass
    let com: Vec3 = tagged.iter()
        .map(|p| Vec3::from(p.position))
        .fold(Vec3::ZERO, |a, b| a + b)
        / tagged.len() as f32;

    let mut prefab = Prefab::new(
        name,
        sim.force_matrix.clone(),
        sim.params.type_count,
    );
    prefab.particle_count = tagged.len();

    for p in tagged {
        prefab.particles.push(PrefabParticle {
            relative_position: (Vec3::from(p.position) - com).into(),
            kind: p.kind,
        });
    }

    sim.book.add_prefab(prefab); // also saves book.json
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn type_color_egui(kind: usize) -> Color32 {
    let colors = [
        Color32::from_rgb(245, 92, 92),
        Color32::from_rgb(92, 194, 245),
        Color32::from_rgb(143, 245, 92),
        Color32::from_rgb(245, 194, 66),
        Color32::from_rgb(220, 117, 245),
        Color32::from_rgb(245, 143, 66),
        Color32::from_rgb(92, 245, 194),
        Color32::from_rgb(245, 194, 220),
    ];
    colors[kind % colors.len()]
}