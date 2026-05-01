use crate::sim::{CpuStepMode, SimState};
use egui::{Color32, Context, Grid, RichText, Slider};
use glam::{Vec3, Vec4Swizzles};

// UI modules
pub mod matrix_ui;
pub mod selection_ui;
pub mod trace_ui;
pub mod debug_ui;
pub mod book_ui;

// ── UiState ───────────────────────────────────────────────────────────────────
pub struct UiState {
    pub paused: bool,
    pub step_once: bool,
    pub use_gpu_physics: bool,
    pub cap_to_bounds: bool,

    // ── Free-fly camera ───────────────────────────────────────────────────────
    pub fly_pos: Vec3,
    pub fly_yaw: f32,
    pub fly_pitch: f32,
    pub fly_speed: f32,
    pub camera_mode: bool,
    pub mouse_look_active: bool,

    // camera_dist is now unused since slice_center is user-controlled

    // ── Book panel ────────────────────────────────────────────────────────────
    pub show_book: bool,
    pub show_matrix_editor: bool,
    pub new_prefab_name: String,

    // ── Selection ─────────────────────────────────────────────────────────────
    pub selected_prefab: Option<usize>,
    pub selected_indices: Vec<usize>,
    pub highlighted_indices: Vec<usize>,
    pub selection_mode: SelectionMode,
    pub brush_radius: f32,
    pub drag_start: Option<egui::Pos2>,
    pub drag_end: Option<egui::Pos2>,
    pub is_selecting: bool,
    pub selection_readback_needed: bool,
    pub gpu_selection_params: crate::renderer::compute::SelectionParams,

    // ── Camera projection matrices (written by renderer each frame) ───────────
    pub view_proj: glam::Mat4,
    pub drag_mode: DragMode,
    pub hover_index: Option<usize>,
    pub viewport: [u32; 2],
    pub view_matrix: glam::Mat4,
    pub slice_center: f32,
    pub slice_thickness: f32,
    pub move_start_mouse: Option<egui::Pos2>,
    pub move_start_positions: Vec<[f32; 3]>,
    pub pending_assign_type: u32,

    // ── Rule matrix hold-click state ──────────────────────────────────────────
    pub matrix_hold: Option<(usize, usize, f32)>,
    pub matrix_hold_timer: f32,

    // ── Trail/trace state ───────────────────────────────────────────────────────
    pub traces: bool, // Java-style simple boolean
    pub trace_len: u32,
    pub trace_fade_alpha: f32,
    pub trace_render_mode: TraceRenderMode,
    pub trace_type_filter: i32,
    pub trace_trigger_only: bool,
    pub debug_trails: bool, // Enable trail debug output

    // Trace Matrix State
    pub trace_show_tools: bool,
    pub trace_paint_value: u32,
    pub trace_drag_paint: bool,
    pub trace_show_indices: bool,
    pub trace_symmetry_lock: bool,
    pub trace_hovered_cell: Option<(usize, usize)>,
    pub trace_brush: TraceBrush,
    pub trace_ui_edit_only: bool,

    // ── Matrix clipboard and presets ───────────────────────────────────────────
    pub force_clipboard: Option<Vec<f32>>,
    pub reaction_clipboard: Option<Vec<i32>>,
    pub trace_clipboard: Option<Vec<u32>>,
    pub show_heat_map: bool,
    pub show_matrix_stats: bool,
    pub force_presets: Vec<String>,
    pub reaction_presets: Vec<String>,
    pub trace_presets: Vec<String>,
    pub active_matrix_tab: ActiveMatrixTab,
    
    // Book and profile fields
    pub selected_creature: Option<usize>,
    pub save_profile_now: bool,
    pub auto_save_profiles: bool,
    pub auto_save_interval: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Rect,
    Brush,
    Slice,
}

#[derive(PartialEq)]
pub enum ActiveMatrixTab {
    Rules,
    Reactions,
    Traces,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceRenderMode {
    Off,
    Simple, // Java-style framebuffer accumulation
    Lines,
    Dots,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceBrush {
    Set,
    Add,
    Subtract,
    Multiply,
    Erase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionOp {
    Replace,
    Add,
    Remove,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragMode {
    None,
    Selecting,
    MovingSelection,
}

impl UiState {
    pub fn new() -> Self {
        Self {
            paused: false,
            step_once: false,
            use_gpu_physics: true,
            cap_to_bounds: true,

            fly_pos: Vec3::new(0.5, 0.5, 2.0),
            fly_yaw: -std::f32::consts::PI / 2.0, // Point toward -Z (center)
            fly_pitch: 0.0,
            fly_speed: 0.5,
            camera_mode: false,
            mouse_look_active: false,

            // camera_dist removed - slice_center is now user-controlled
            show_book: false,
            show_matrix_editor: false,
            new_prefab_name: String::new(),
            selected_prefab: None,

            selected_indices: Vec::new(),
            highlighted_indices: Vec::new(),
            selection_mode: SelectionMode::Rect,
            brush_radius: 40.0,
            drag_start: None,
            drag_end: None,
            is_selecting: false,
            selection_readback_needed: false,
            view_proj: glam::Mat4::IDENTITY,
            gpu_selection_params: crate::renderer::compute::SelectionParams {
                mode_flags: [0, 0, 0, 0],     // mode + flags/padding
                rect_min: [0.0, 0.0, 0.0, 0.0], // rect_min + padding
                rect_max: [0.0, 0.0, 0.0, 0.0], // rect_max + padding
                brush_data: [0.0, 0.0, 40.0, 0.0], // brush_center.x, brush_center.y, brush_radius, slice_depth
            },

            drag_mode: DragMode::None,
            hover_index: None,
            viewport: [1, 1],
            view_matrix: glam::Mat4::IDENTITY,
            slice_center: 0.0,
            slice_thickness: 0.2,
            move_start_mouse: None,
            move_start_positions: Vec::new(),
            pending_assign_type: 0,

            matrix_hold: None,
            matrix_hold_timer: 0.0,

            traces: false, // Java-style: start with traces off
            trace_len: 16,
            trace_fade_alpha: 0.98,
            trace_render_mode: TraceRenderMode::Off,
            trace_type_filter: -1,
            trace_trigger_only: false,
            debug_trails: false, // Debug output off by default

            // Trace Matrix State
            trace_show_tools: true,
            trace_paint_value: 30,
            trace_drag_paint: false,
            trace_show_indices: true,
            trace_symmetry_lock: false,
            trace_hovered_cell: None,
            trace_brush: TraceBrush::Set,
            trace_ui_edit_only: false,

            // Matrix clipboard and presets
            force_clipboard: None,
            reaction_clipboard: None,
            trace_clipboard: None,
            show_heat_map: false,
            show_matrix_stats: false,
            force_presets: Vec::new(),
            reaction_presets: Vec::new(),
            trace_presets: Vec::new(),
            active_matrix_tab: ActiveMatrixTab::Rules,
            
            // Book and profile fields
            selected_creature: None,
            save_profile_now: false,
            auto_save_profiles: false,
            auto_save_interval: 500,
        }
    }

    /// Forward direction in world space from current yaw/pitch
    pub fn fly_forward(&self) -> Vec3 {
        Vec3::new(
            self.fly_yaw.cos() * self.fly_pitch.cos(),
            self.fly_pitch.sin(),
            self.fly_yaw.sin() * self.fly_pitch.cos(),
        )
        .normalize()
    }

    /// Right vector (perpendicular to forward, in the XZ plane)
    pub fn fly_right(&self) -> Vec3 {
        self.fly_forward().cross(Vec3::Y).normalize()
    }

    #[allow(dead_code)]
    pub fn world_to_screen(&self, pos: Vec3, viewport: [u32; 2]) -> Option<egui::Pos2> {
        let clip = self.view_proj * pos.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.xyz() / clip.w;
        Some(egui::pos2(
            (ndc.x * 0.5 + 0.5) * viewport[0] as f32,
            (1.0 - (ndc.y * 0.5 + 0.5)) * viewport[1] as f32,
        ))
    }
}

// ── Main UI entry point ───────────────────────────────────────────────────────
pub fn draw_ui(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    draw_controls(ctx, sim, ui);
    if ui.show_matrix_editor {
        draw_tabbed_matrices(ctx, sim, ui);
    }
    if ui.show_book {
        draw_book(ctx, sim, ui);
    }
    handle_input(ctx, sim, ui);
}

// ── Tabbed Matrix Editor ───────────────────────────────────────────────────────
fn draw_tabbed_matrices(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    egui::Window::new("Matrix Editor")
        .default_pos([350.0, 12.0])
        .resizable(true)
        .default_width(600.0)
        .show(ctx, |e| {
            // Tab selection
            e.horizontal(|e| {
                e.selectable_value(&mut ui.active_matrix_tab, ActiveMatrixTab::Rules, "Rules");
                e.selectable_value(
                    &mut ui.active_matrix_tab,
                    ActiveMatrixTab::Reactions,
                    "Reactions",
                );
                e.selectable_value(&mut ui.active_matrix_tab, ActiveMatrixTab::Traces, "Traces");
            });
            e.separator();

            // Show content based on active tab
            match ui.active_matrix_tab {
                ActiveMatrixTab::Rules => draw_rule_matrix_content(ctx, sim, ui, e),
                ActiveMatrixTab::Reactions => draw_reaction_matrix_content(ctx, sim, ui, e),
                ActiveMatrixTab::Traces => draw_trace_matrix_content(ctx, sim, ui, e),
            }
        });
}

// ── Selection helpers ─────────────────────────────────────────────────────────
fn apply_selection_op(selected: &mut Vec<usize>, hits: &[usize], op: SelectionOp) {
    match op {
        SelectionOp::Replace => {
            selected.clear();
            selected.extend_from_slice(hits);
        }
        SelectionOp::Add => {
            selected.extend_from_slice(hits);
        }
        SelectionOp::Remove => {
            let rm: std::collections::HashSet<_> = hits.iter().copied().collect();
            selected.retain(|i| !rm.contains(i));
        }
    }
    dedup_selection(selected);
}

fn dedup_selection(v: &mut Vec<usize>) {
    v.sort_unstable();
    v.dedup();
}

fn set_trace_cell(
    sim: &mut crate::sim::SimState,
    row: usize,
    col: usize,
    value: u32,
    symmetry_lock: bool,
) {
    let n = sim.params.type_count as usize;
    if row >= n || col >= n {
        return;
    }

    let idx = row * n + col;
    if idx < sim.trace_len_matrix.len() {
        sim.trace_len_matrix[idx] = value;
        sim.trace_len_matrix_dirty = true;
    }

    if symmetry_lock && row != col {
        let mirror_idx = col * n + row;
        if mirror_idx < sim.trace_len_matrix.len() {
            sim.trace_len_matrix[mirror_idx] = value;
            sim.trace_len_matrix_dirty = true;
        }
    }
}

fn draw_trace_matrix_ui(
    ui: &mut egui::Ui,
    state: &mut UiState,
    sim: &mut crate::sim::SimState,
) {
    let n = sim.params.type_count as usize;

    ui.horizontal(|ui| {
        egui::ComboBox::from_label("Render")
            .selected_text(format!("{:?}", state.trace_render_mode))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.trace_render_mode, TraceRenderMode::Off, "Off");
                ui.selectable_value(&mut state.trace_render_mode, TraceRenderMode::Lines, "Lines");
                ui.selectable_value(&mut state.trace_render_mode, TraceRenderMode::Dots, "Dots");
            });

        ui.add(egui::Slider::new(&mut state.trace_len, 1..=20).text("history"));
        ui.add(egui::Slider::new(&mut state.trace_fade_alpha, 0.0..=1.0).text("fade"));
    });

    ui.separator();

    ui.collapsing("Trace Matrix Tools", |ui| {
        ui.horizontal(|ui| {
            ui.checkbox(&mut state.trace_show_tools, "show tools");
            ui.checkbox(&mut state.trace_show_indices, "indices");
            ui.checkbox(&mut state.trace_symmetry_lock, "symmetry");
            ui.checkbox(&mut state.trace_drag_paint, "drag paint");
        });

        ui.horizontal(|ui| {
            ui.label("Brush:");
            egui::ComboBox::from_label("")
                .selected_text(format!("{:?}", state.trace_brush))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.trace_brush, TraceBrush::Set, "Set");
                    ui.selectable_value(&mut state.trace_brush, TraceBrush::Add, "Add");
                    ui.selectable_value(&mut state.trace_brush, TraceBrush::Subtract, "Subtract");
                    ui.selectable_value(&mut state.trace_brush, TraceBrush::Multiply, "Multiply");
                    ui.selectable_value(&mut state.trace_brush, TraceBrush::Erase, "Erase");
                });

            ui.label("Paint lifetime:");
            ui.add(egui::DragValue::new(&mut state.trace_paint_value)
                .speed(1)
                .clamp_range(0..=600));

            if ui.button("Clear all").clicked() {
                for v in &mut sim.trace_len_matrix {
                    *v = 0;
                }
                sim.trace_len_matrix_dirty = true;
            }

            if ui.button("Fill all").clicked() {
                for v in &mut sim.trace_len_matrix {
                    *v = state.trace_paint_value;
                }
                sim.trace_len_matrix_dirty = true;
            }

            if ui.button("Diag only").clicked() {
                for r in 0..n {
                    for c in 0..n {
                        sim.trace_len_matrix[r * n + c] =
                            if r == c { state.trace_paint_value } else { 0 };
                    }
                }
                sim.trace_len_matrix_dirty = true;
            }

            if ui.button("From reactions").clicked() {
                for r in 0..n {
                    for c in 0..n {
                        let reaction_idx = r * n + c;
                        let trace_idx = r * n + c;
                        if trace_idx < sim.trace_len_matrix.len() && reaction_idx < sim.reaction_table.len() {
                            sim.trace_len_matrix[trace_idx] = if sim.reaction_table[reaction_idx] > 0 { state.trace_paint_value } else { 0 };
                        }
                    }
                }
                sim.trace_len_matrix_dirty = true;
            }
        });

        let active = sim.trace_len_matrix.iter().filter(|&&v| v > 0).count();
        ui.label(format!(
            "Active trace cells: {}/{} | dirty: {} | hovered: {:?}",
            active,
            n * n,
            sim.trace_len_matrix_dirty,
            state.trace_hovered_cell
        ));
    });

    ui.separator();

    // Debug Panel
    ui.collapsing("Trace Debug", |ui| {
        ui.label(format!("trace_len_matrix_dirty: {}", sim.trace_len_matrix_dirty));
        ui.label(format!("active cells: {}", sim.trace_len_matrix.iter().filter(|&&v| v > 0).count()));
        ui.label(format!("matrix size: {}x{}", n, n));
        ui.label(format!("trace_len: {}", state.trace_len));
        ui.label(format!("trigger_only: {}", state.trace_trigger_only));
        
        // Add more debug info as available
        if let Some((r, c)) = state.trace_hovered_cell {
            let idx = r * n + c;
            if idx < sim.trace_len_matrix.len() {
                ui.label(format!("hovered cell ({},{}) value: {}", r, c, sim.trace_len_matrix[idx]));
            }
        }
    });

    ui.separator();

    if sim.trace_len_matrix.len() != n * n {
        sim.trace_len_matrix.resize(n * n, 0);
        sim.trace_len_matrix_dirty = true;
    }

    // Find max value for heatmap normalization
    let max_value = sim.trace_len_matrix.iter().copied().max().unwrap_or(1).max(1);

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .max_height(420.0)
        .show(ui, |ui| {
            egui::Grid::new("trace_matrix_grid")
                .spacing([2.0, 2.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label("");

                    for c in 0..n {
                        if state.trace_show_indices {
                            ui.label(format!("{c}"));
                        } else {
                            ui.label("");
                        }
                    }
                    ui.end_row();

                    for r in 0..n {
                        if state.trace_show_indices {
                            ui.label(format!("{r}"));
                        } else {
                            ui.label("");
                        }

                        for c in 0..n {
                            let idx = r * n + c;
                            let value = sim.trace_len_matrix[idx];

                            // Heatmap coloring
                            let t = (value as f32 / max_value as f32).clamp(0.0, 1.0);
                            let color = if value == 0 {
                                egui::Color32::from_rgb(40, 40, 40) // Dark for zero
                            } else {
                                egui::Color32::from_rgb(
                                    (t * 255.0) as u8,           // Red channel
                                    ((1.0 - (t - 0.5).abs() * 2.0) * 255.0) as u8, // Green channel (peaks at middle)
                                    ((1.0 - t) * 255.0) as u8,    // Blue channel
                                )
                            };

                            let button_text = if value == 0 {
                                ".".to_string()
                            } else {
                                value.to_string()
                            };

                            let response = ui.add_sized(
                                [34.0, 24.0],
                                egui::Button::new(button_text)
                                    .fill(color),
                            );

                            if response.hovered() {
                                state.trace_hovered_cell = Some((r, c));
                            }

                            if response.clicked() {
                                let new_value = match state.trace_brush {
                                    TraceBrush::Set => {
                                        if value == state.trace_paint_value { 0 } else { state.trace_paint_value }
                                    },
                                    TraceBrush::Add => value.saturating_add(state.trace_paint_value),
                                    TraceBrush::Subtract => value.saturating_sub(state.trace_paint_value),
                                    TraceBrush::Multiply => value.saturating_mul(state.trace_paint_value),
                                    TraceBrush::Erase => 0,
                                };

                                set_trace_cell(
                                    sim,
                                    r,
                                    c,
                                    new_value,
                                    state.trace_symmetry_lock,
                                );
                            }

                            if state.trace_drag_paint
                                && response.hovered()
                                && ui.input(|i| i.pointer.primary_down())
                            {
                                let new_value = match state.trace_brush {
                                    TraceBrush::Set => state.trace_paint_value,
                                    TraceBrush::Add => value.saturating_add(state.trace_paint_value),
                                    TraceBrush::Subtract => value.saturating_sub(state.trace_paint_value),
                                    TraceBrush::Multiply => value.saturating_mul(state.trace_paint_value),
                                    TraceBrush::Erase => 0,
                                };

                                set_trace_cell(
                                    sim,
                                    r,
                                    c,
                                    new_value,
                                    state.trace_symmetry_lock,
                                );
                            }

                            if response.secondary_clicked() {
                                set_trace_cell(
                                    sim,
                                    r,
                                    c,
                                    0,
                                    state.trace_symmetry_lock,
                                );
                            }

                            response.on_hover_text(format!(
                                "Trace {} -> {}\nLifetime: {}\nBrush: {:?}\nLeft click: apply brush\nRight click: clear",
                                r, c, value, state.trace_brush
                            ));
                        }

                        ui.end_row();
                    }
                });
        });
}

// ── Combined input handler ────────────────────────────────────────────────────
// Camera-mode WASD always works even over UI panels.
// Mouse-look, scroll, and selection only fire when pointer is in the viewport.
fn handle_input(ctx: &egui::Context, sim: &mut SimState, ui: &mut UiState) {
    let dt = ctx.input(|i| i.stable_dt);

    // Toggle trail with T key - cycle through functional modes (skip Simple)
    if ctx.input(|i| i.key_pressed(egui::Key::T)) {
        ui.trace_render_mode = match ui.trace_render_mode {
            TraceRenderMode::Off => TraceRenderMode::Dots,
            TraceRenderMode::Simple => TraceRenderMode::Dots,
            TraceRenderMode::Dots => TraceRenderMode::Lines,
            TraceRenderMode::Lines => TraceRenderMode::Off,
        };
    }

    // ── CAMERA MODE ───────────────────────────────────────────────────────────
    if ui.camera_mode {
        // Keyboard movement — unconditional, works even when cursor is on a panel
        let fwd = ui.fly_forward();
        let right = ui.fly_right();
        let up = Vec3::Y;
        let speed = ui.fly_speed;

        ctx.input(|i| {
            use egui::Key;
            if i.key_down(Key::W) {
                ui.fly_pos += fwd * speed * dt;
            }
            if i.key_down(Key::S) {
                ui.fly_pos -= fwd * speed * dt;
            }
            if i.key_down(Key::D) {
                ui.fly_pos += right * speed * dt;
            }
            if i.key_down(Key::A) {
                ui.fly_pos -= right * speed * dt;
            }
            if i.key_down(Key::E) {
                ui.fly_pos += up * speed * dt;
            }
            if i.key_down(Key::Q) {
                ui.fly_pos -= up * speed * dt;
            }
        });

        // Mouse-look + scroll — only when NOT over a panel
        if !ctx.is_pointer_over_area() {
            let pointer = ctx.input(|i| i.pointer.clone());

            if pointer.button_pressed(egui::PointerButton::Secondary) {
                ui.mouse_look_active = true;
            }
            if pointer.button_released(egui::PointerButton::Secondary) {
                ui.mouse_look_active = false;
            }
            if ui.mouse_look_active && pointer.button_down(egui::PointerButton::Secondary) {
                let delta = ctx.input(|i| i.pointer.delta());
                ui.fly_yaw += delta.x * 0.003;
                ui.fly_pitch = (ui.fly_pitch - delta.y * 0.003)
                    .clamp(-89_f32.to_radians(), 89_f32.to_radians());
            }

            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                ui.fly_speed = (ui.fly_speed * (1.0 + scroll * 0.05)).clamp(0.001, 50.0);
            }
        }

        return;
    }

    // ── SELECTION MODE — only outside UI panels ───────────────────────────────
    if ctx.is_pointer_over_area() {
        ui.hover_index = None;
        if ui.drag_mode == DragMode::None {
            ui.highlighted_indices.clear();
        }
        return;
    }

    let pointer = ctx.input(|i| i.pointer.clone());
    let latest = pointer.latest_pos();

    // Active move-drag takes full control
    if ui.drag_mode == DragMode::MovingSelection {
        handle_move_drag(pointer, latest, sim, ui);
        return;
    }

    // Hover preview when idle - clear GPU selection when not interacting
    if ui.drag_mode == DragMode::None && !pointer.any_down() {
        ui.gpu_selection_params.mode_flags[0] = 0; // No selection
        ui.hover_index = None;
        ui.highlighted_indices.clear();
    }

    if pointer.primary_pressed() {
        if let Some(pos) = pointer.press_origin() {
            ui.drag_start = Some(pos);
            ui.drag_end = Some(pos);
            ui.drag_mode = DragMode::Selecting;
            ui.is_selecting = true;

            // Set GPU selection parameters based on mode
            match ui.selection_mode {
                SelectionMode::Rect => {
                    ui.gpu_selection_params.mode_flags[0] = 1; // Rect mode
                    ui.gpu_selection_params.rect_min = [pos.x, pos.y, 0.0, 0.0];
                    ui.gpu_selection_params.rect_max = [pos.x, pos.y, 0.0, 0.0];
                }
                SelectionMode::Brush => {
                    ui.gpu_selection_params.mode_flags[0] = 2; // Brush mode
                    ui.gpu_selection_params.brush_data = [pos.x, pos.y, ui.brush_radius, 0.0];
                }
                SelectionMode::Slice => {
                    ui.gpu_selection_params.mode_flags[0] = 3; // Slice mode
                    ui.gpu_selection_params.brush_data = [0.0, 0.0, 0.0, ui.slice_center];
                }
            }
        }
    }

    if pointer.primary_down() {
        if let Some(pos) = latest {
            ui.drag_end = Some(pos);
            match ui.selection_mode {
                SelectionMode::Rect => {
                    if let Some(start) = ui.drag_start {
                        let rect = egui::Rect::from_two_pos(start, pos);
                        ui.gpu_selection_params.rect_min = [rect.min.x, rect.min.y, 0.0, 0.0];
                        ui.gpu_selection_params.rect_max = [rect.max.x, rect.max.y, 0.0, 0.0];
                    }
                }
                SelectionMode::Brush => {
                    ui.gpu_selection_params.brush_data = [pos.x, pos.y, ui.brush_radius, 0.0];
                }
                SelectionMode::Slice => {
                    ui.gpu_selection_params.brush_data = [0.0, 0.0, 0.0, ui.slice_center];
                }
            }
        }
    }

    if pointer.primary_released() {
        // Keep selection active - GPU will continue to highlight
        ui.drag_mode = DragMode::None;
        ui.drag_start = None;
        ui.drag_end = None;
        ui.is_selecting = false;

        // Trigger selection readback for all selection modes
        ui.selection_readback_needed = true;
    }

    // Middle-mouse → move selection on camera plane
    if pointer.button_pressed(egui::PointerButton::Middle) && !ui.selected_indices.is_empty() {
        if let Some(pos) = pointer.press_origin() {
            ui.drag_mode = DragMode::MovingSelection;
            ui.move_start_mouse = Some(pos);
            ui.move_start_positions = ui
                .selected_indices
                .iter()
                .filter_map(|&i| sim.particles.get(i).map(|p| p.position))
                .collect();
        }
    }
}

fn handle_move_drag(
    pointer: egui::PointerState,
    latest: Option<egui::Pos2>,
    sim: &mut SimState,
    ui: &mut UiState,
) {
    if pointer.button_down(egui::PointerButton::Middle) {
        if let (Some(start), Some(now)) = (ui.move_start_mouse, latest) {
            let dx = now.x - start.x;
            let dy = now.y - start.y;
            let (right, up, _) = crate::selection::camera_plane_axes(ui.view_matrix);
            let sx = sim.params.bounds / ui.viewport[0] as f32;
            let sy = sim.params.bounds / ui.viewport[1] as f32;
            let delta = right * (dx * sx) + up * (-dy * sy);

            for (slot, &idx) in ui.selected_indices.iter().enumerate() {
                if let Some(p) = sim.particles.get_mut(idx) {
                    let start_pos = glam::Vec3::from(ui.move_start_positions[slot]);
                    p.position = (start_pos + delta).into();
                }
            }
            sim.particles_dirty = true;
        }
    }
    if pointer.any_released() {
        ui.drag_mode = DragMode::None;
        ui.move_start_mouse = None;
        ui.move_start_positions.clear();
    }
}

// ── Controls panel ────────────────────────────────────────────────────────────
fn draw_controls(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    egui::Window::new("Controls")
        .default_pos([12.0, 12.0])
        .default_width(260.0)
        .resizable(false)
        .show(ctx, |e| {
            e.label(format!("Particles: {}", sim.particles.len()));
            e.label(format!("Types:     {}", sim.params.type_count));
            e.label(format!("Steps:     {}", sim.step_count));
            e.separator();

            e.horizontal(|e| {
                let lbl = if ui.paused { "▶ Resume" } else { "⏸ Pause" };
                if e.button(lbl).clicked() {
                    ui.paused = !ui.paused;
                }
                if e.button("⏭ Step").clicked() {
                    ui.step_once = true;
                }
            });
            e.separator();

            e.checkbox(&mut ui.use_gpu_physics, "Use GPU Physics");
            e.separator();

            e.label("Physics");
            let mut changed = false;

            let mut type_count = sim.params.type_count as f32;
            if e.add(
                Slider::new(
                    &mut type_count,
                    1.0..=crate::renderer::compute::MAX_TYPES as f32,
                )
                .text("types"),
            )
            .changed()
            {
                let nc = type_count as usize;
                if nc != sim.params.type_count {
                    sim.set_type_count(nc);
                }
            }

            let r_max_max = if ui.cap_to_bounds {
                sim.params.bounds
            } else {
                20.0
            };
            changed |= e
                .add(Slider::new(&mut sim.params.r_max, 0.001..=r_max_max).text("r_max"))
                .changed();
            changed |= e
                .add(Slider::new(&mut sim.params.force_scale, 0.0..=20.0).text("force scale"))
                .changed();
            changed |= e
                .add(Slider::new(&mut sim.params.friction, 0.0..=1.0).text("friction"))
                .changed();
            changed |= e
                .add(Slider::new(&mut sim.params.dt, 0.0001..=0.1).text("dt"))
                .changed();
            let max_speed_max = if ui.cap_to_bounds {
                sim.params.bounds
            } else {
                20.0
            };
            changed |= e
                .add(Slider::new(&mut sim.params.max_speed, 0.01..=max_speed_max).text("max speed"))
                .changed();
            changed |= e
                .add(Slider::new(&mut sim.params.beta, 0.01..=0.99).text("beta"))
                .changed();
            changed |= e
                .add(Slider::new(&mut sim.params.particle_size, 0.001..=0.2).text("particle size"))
                .changed();
            changed |= e
                .add(Slider::new(&mut sim.params.bounds, 0.1..=20.0).text("bounds"))
                .changed();
            changed |= e
                .checkbox(&mut ui.cap_to_bounds, "Cap sliders to bounds")
                .changed();
            changed |= e.checkbox(&mut sim.params.wrap, "Wrap").changed();

            e.separator();
            e.checkbox(&mut ui.show_matrix_editor, "Matrix Editor");
            e.separator();
            e.label("CPU step mode");
            e.horizontal(|e| {
                let is_auto = matches!(sim.params.cpu_step_mode, CpuStepMode::Auto);
                let is_naive = matches!(sim.params.cpu_step_mode, CpuStepMode::Naive);
                let is_grid = matches!(sim.params.cpu_step_mode, CpuStepMode::GridExact);

                if e.selectable_label(is_auto, "Auto").clicked() {
                    sim.params.cpu_step_mode = CpuStepMode::Auto;
                    changed = true;
                }
                if e.selectable_label(is_naive, "Naive").clicked() {
                    sim.params.cpu_step_mode = CpuStepMode::Naive;
                    changed = true;
                }
                if e.selectable_label(is_grid, "GridExact").clicked() {
                    sim.params.cpu_step_mode = CpuStepMode::GridExact;
                    changed = true;
                }
            });

            let mut auto_threshold = sim.params.auto_grid_threshold as u32;
            if e.add(Slider::new(&mut auto_threshold, 0..=50_000).text("grid threshold"))
                .changed()
            {
                sim.params.auto_grid_threshold = auto_threshold as usize;
                changed = true;
            }

            let resolved_mode = if sim.last_step_used_grid {
                "GridExact"
            } else {
                "Naive"
            };
            e.label(format!("Last resolved mode: {}", resolved_mode));
            e.label(format!("Neighbor checks: {}", sim.last_neighbor_checks));
            e.label(format!("Grid resolution: {}", sim.last_grid_res));
            e.label(format!("Last step time: {:.3} ms", sim.last_step_ms));
            e.label(format!("Avg step time: {:.3} ms", sim.avg_step_ms));

            if changed {
                sim.params_dirty = true;
            }
            e.separator();

            e.label("Spawn");
            e.horizontal(|e| {
                if e.button("-1k").clicked() {
                    let n = sim.particles.len().saturating_sub(1000);
                    sim.particles.truncate(n);
                    sim.particles_dirty = true;
                }
                if e.button("-100").clicked() {
                    let n = sim.particles.len().saturating_sub(100);
                    sim.particles.truncate(n);
                    sim.particles_dirty = true;
                }
                if e.button("+100").clicked() {
                    sim.spawn_random(100);
                }
                if e.button("+1k").clicked() {
                    sim.spawn_random(1000);
                }
                if e.button("+10k").clicked() {
                    sim.spawn_random(10000);
                }
            });
            e.horizontal(|e| {
                if e.button("Clear").clicked() {
                    sim.clear_particles();
                }
                if e.button("🎲 Rules").clicked() {
                    sim.randomize_rules();
                }
            });
            e.separator();

            e.label("Camera");
            let cam_label = if ui.camera_mode {
                "🎥 Camera Mode  [ON]"
            } else {
                "🎥 Camera Mode [OFF]"
            };
            if e.button(cam_label).clicked() {
                ui.camera_mode = !ui.camera_mode;
                ui.mouse_look_active = false;
            }

            if ui.camera_mode {
                e.add(
                    Slider::new(&mut ui.fly_speed, 0.001..=50.0)
                        .logarithmic(true)
                        .text("fly speed"),
                );
                e.label("W/S = fwd/back  A/D = strafe\nQ/E = down/up   RMB drag = look");
                if e.button("Reset pos").clicked() {
                    ui.fly_pos = Vec3::new(
                        sim.params.bounds * 0.5,
                        sim.params.bounds * 0.5,
                        sim.params.bounds * 2.0,
                    );
                    // Look at center from current position
                    let center = Vec3::splat(sim.params.bounds * 0.5);
                    let forward = (center - ui.fly_pos).normalize();
                    ui.fly_yaw = forward.x.atan2(forward.z);
                    ui.fly_pitch = forward.y.asin();
                }
                // Expose raw angles so player can nudge them precisely
                e.add(Slider::new(&mut ui.fly_yaw, -3.14..=3.14).text("yaw"));
                e.add(Slider::new(&mut ui.fly_pitch, -1.5..=1.5).text("pitch"));
            }
            e.separator();

            e.label("Selection");
            e.horizontal(|e| {
                if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Rect), "Rect")
                    .clicked()
                {
                    ui.selection_mode = SelectionMode::Rect;
                }
                if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Brush), "Brush")
                    .clicked()
                {
                    ui.selection_mode = SelectionMode::Brush;
                }
                if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Slice), "Slice")
                    .clicked()
                {
                    ui.selection_mode = SelectionMode::Slice;
                }
            });
            if matches!(ui.selection_mode, SelectionMode::Slice) {
                e.add(Slider::new(&mut ui.slice_center, -20.0..=20.0).text("slice center"));
                e.add(Slider::new(&mut ui.slice_thickness, 0.001..=5.0).text("slice thickness"));
            }
            if matches!(ui.selection_mode, SelectionMode::Brush) {
                e.add(Slider::new(&mut ui.brush_radius, 2.0..=300.0).text("brush radius"));
            }

            e.label(format!("{} selected", ui.selected_indices.len()));
            if !ui.selected_indices.is_empty() {
                e.horizontal(|e| {
                    if e.button("Clear sel").clicked() {
                        ui.selected_indices.clear();
                        ui.highlighted_indices.clear();
                    }
                    if e.button("Delete").clicked() {
                        sim.delete_particles(&ui.selected_indices);
                        ui.selected_indices.clear();
                    }
                    if e.button("Duplicate").clicked() {
                        sim.duplicate_particles(&ui.selected_indices);
                    }
                });
                e.horizontal(|e| {
                    if e.button("Cool down").clicked() {
                        sim.scale_velocities(0.1);
                    }
                    if e.button("Energy boost").clicked() {
                        sim.scale_velocities(2.0);
                    }
                    if e.button("Freeze").clicked() {
                        sim.scale_velocities(0.0);
                    }
                });
                e.horizontal(|e| {
                    e.label("Assign type:");
                    let mut tv = ui.pending_assign_type as usize;
                    if e.add(
                        Slider::new(&mut tv, 0..=sim.params.type_count.saturating_sub(1)).text(""),
                    )
                    .changed()
                    {
                        ui.pending_assign_type = tv as u32;
                    }
                    if e.button("Assign").clicked() {
                        sim.assign_type_to_particles(&ui.selected_indices, ui.pending_assign_type);
                    }
                });
            }
            e.separator();

            if e.button("📖 Creature Book").clicked() {
                ui.show_book = !ui.show_book;
            }
        });
}

// ── Matrix preset functions ───────────────────────────────────────────────────
fn apply_force_preset(sim: &mut SimState, preset: &str) {
    let n = sim.params.type_count;
    match preset {
        "Attractor-Repulsor" => {
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        sim.set_rule(i, j, -0.8); // Self-repulsion
                    } else if i < j {
                        sim.set_rule(i, j, 0.6); // Attract
                        sim.set_rule(j, i, -0.4); // Repel
                    }
                }
            }
        }
        "Circular Flow" => {
            for i in 0..n {
                for j in 0..n {
                    let angle = (i as f32 - j as f32) * 2.0 * std::f32::consts::PI / n as f32;
                    sim.set_rule(i, j, angle.cos() * 0.7);
                }
            }
        }
        "Cluster Formation" => {
            for i in 0..n {
                for j in 0..n {
                    let dist = (i as f32 - j as f32).abs() / n as f32;
                    sim.set_rule(i, j, if dist < 0.3 { 0.8 } else { -0.6 });
                }
            }
        }
        "Wave Pattern" => {
            for i in 0..n {
                for j in 0..n {
                    let wave = ((i + j) as f32 * std::f32::consts::PI / n as f32).sin();
                    sim.set_rule(i, j, wave * 0.9);
                }
            }
        }
        "Chaotic Dance" => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for i in 0..n {
                for j in 0..n {
                    sim.set_rule(i, j, rng.gen_range(-1.0..1.0));
                }
            }
        }
        _ => {}
    }
}

// ── Matrix content functions ─────────────────────────────────────────────────────
fn draw_rule_matrix_content(ctx: &Context, sim: &mut SimState, ui: &mut UiState, e: &mut egui::Ui) {
    let n = sim.params.type_count;
    let dt = ctx.input(|i| i.stable_dt);

    // Advance hold-click ramp (150 ms delay, then 1.0 unit/s)
    if let Some((row, col, sign)) = ui.matrix_hold {
        ui.matrix_hold_timer += dt;
        if ui.matrix_hold_timer > 0.15 {
            let val = sim.get_rule(row, col);
            sim.set_rule(row, col, (val + sign * dt).clamp(-1.0, 1.0));
        }
    }

    e.label("LMB = +  RMB = −  hold to ramp  scroll = fine");
    e.separator();

    // Enhanced controls row
    e.horizontal(|e| {
        if e.button("Zero all").clicked() {
            for i in 0..n {
                for j in 0..n {
                    sim.set_rule(i, j, 0.0);
                }
            }
        }
        if e.button("Symmetrize").clicked() {
            for i in 0..n {
                for j in (i + 1)..n {
                    let avg = (sim.get_rule(i, j) + sim.get_rule(j, i)) * 0.5;
                    sim.set_rule(i, j, avg);
                    sim.set_rule(j, i, avg);
                }
            }
        }
        if e.button("Invert").clicked() {
            for i in 0..n {
                for j in 0..n {
                    let v = sim.get_rule(i, j);
                    sim.set_rule(i, j, -v);
                }
            }
        }
    });

    // Copy/Paste and Presets row
    e.horizontal(|e| {
        if e.button("Copy").clicked() {
            let mut matrix = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    matrix.push(sim.get_rule(i, j));
                }
            }
            ui.force_clipboard = Some(matrix);
        }
        if e.button("Paste").clicked() {
            if let Some(ref matrix) = ui.force_clipboard {
                if matrix.len() == n * n {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_rule(i, j, matrix[i * n + j]);
                        }
                    }
                }
            }
        }
        if e.button("Random").clicked() {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for i in 0..n {
                for j in 0..n {
                    sim.set_rule(i, j, rng.gen_range(-1.0..1.0));
                }
            }
        }
        e.checkbox(&mut ui.show_heat_map, "Heat Map");
        e.checkbox(&mut ui.show_matrix_stats, "Stats");
    });

    // Presets dropdown
    e.horizontal(|e| {
        egui::ComboBox::from_label("Presets")
            .selected_text("Select preset...")
            .show_ui(e, |e| {
                if ui.force_presets.is_empty() {
                    ui.force_presets = vec![
                        "Attractor-Repulsor".to_string(),
                        "Circular Flow".to_string(),
                        "Cluster Formation".to_string(),
                        "Wave Pattern".to_string(),
                        "Chaotic Dance".to_string(),
                    ];
                }
                for preset in &ui.force_presets {
                    if e.selectable_label(false, preset).clicked() {
                        apply_force_preset(sim, preset);
                    }
                }
            });
    });
    e.separator();

    // Add scroll area so large matrices don't break window layout
    if n > 16 {
        e.colored_label(
            Color32::YELLOW,
            format!("Matrix hidden: {}x{} is too expensive for live UI. Reduce types or use presets/tools.", n, n)
        );
    } else {
        egui::ScrollArea::both().show(e, |e| {
            Grid::new("matrix").spacing([2.0, 2.0]).show(e, |e| {
                e.label("");
                for col in 0..n {
                    e.label(RichText::new(format!("T{}", col)).color(type_color_egui(col)));
                }
                e.end_row();

                for row in 0..n {
                    e.label(RichText::new(format!("T{}", row)).color(type_color_egui(row)));

                    for col in 0..n {
                        let val = sim.get_rule(row, col);

                        let cell_color = if val > 0.0 {
                            Color32::from_rgb(30, (40.0 + val * 180.0) as u8, 30)
                        } else if val < 0.0 {
                            Color32::from_rgb((40.0 + val.abs() * 180.0) as u8, 30, 30)
                        } else {
                            Color32::from_rgb(40, 40, 40)
                        };

                        let resp = e.add(
                            egui::Button::new(RichText::new(format!("{:.2}", val)).size(11.0))
                                .fill(cell_color)
                                .min_size(egui::vec2(40.0, 28.0)),
                        );

                        // Single click: ±0.1
                        if resp.clicked() {
                            sim.set_rule(row, col, (val + 0.1).clamp(-1.0, 1.0));
                        }
                        if resp.secondary_clicked() {
                            sim.set_rule(row, col, (val - 0.1).clamp(-1.0, 1.0));
                        }

                        // Hold tracking
                        if resp.is_pointer_button_down_on() {
                            let sign = if ctx.input(|i| i.pointer.secondary_down()) {
                                -1.0_f32
                            } else {
                                1.0_f32
                            };
                            if ui.matrix_hold != Some((row, col, sign)) {
                                ui.matrix_hold = Some((row, col, sign));
                                ui.matrix_hold_timer = 0.0;
                            }
                        } else if matches!(ui.matrix_hold, Some((r, c, _)) if r == row && c == col)
                        {
                            ui.matrix_hold = None;
                            ui.matrix_hold_timer = 0.0;
                        }

                        // Scroll fine-tune
                        if resp.hovered() {
                            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                            if scroll != 0.0 {
                                sim.set_rule(row, col, (val + scroll * 0.005).clamp(-1.0, 1.0));
                            }
                        }
                    }
                    e.end_row();
                }
            });
        });
    }

    e.separator();

    // Statistics panel
    if ui.show_matrix_stats {
        e.horizontal(|e| {
            let mut sum = 0.0;
            let mut positive_count = 0;
            let mut negative_count = 0;
            let mut max_val: f32 = 0.0;
            let mut min_val: f32 = 0.0;

            for i in 0..n {
                for j in 0..n {
                    let val = sim.get_rule(i, j);
                    sum += val;
                    if val > 0.0 {
                        positive_count += 1;
                    }
                    if val < 0.0 {
                        negative_count += 1;
                    }
                    max_val = max_val.max(val);
                    min_val = min_val.min(val);
                }
            }

            e.label(format!("Sum: {:.2}", sum));
            e.label(format!("Avg: {:.2}", sum / (n * n) as f32));
            e.label(format!("+/−: {positive_count}/{negative_count}"));
            e.label(format!("Range: [{min_val:.2}, {max_val:.2}]"));
        });
    }

    e.horizontal(|e| {
        e.colored_label(Color32::from_rgb(30, 220, 30), "■ attract");
        e.colored_label(Color32::from_rgb(220, 30, 30), "■ repel");
    });
}

fn draw_reaction_matrix_content(
    ctx: &Context,
    sim: &mut SimState,
    ui: &mut UiState,
    e: &mut egui::Ui,
) {
    let n = sim.params.type_count;
    let _dt = ctx.input(|i| i.stable_dt);

    e.label("LMB = cycle result  RMB = no reaction");
    e.separator();

    // Reaction controls
    e.horizontal(|e| {
        let mut enabled = sim.params.reactions_enabled;
        if e.checkbox(&mut enabled, "Enable Reactions").changed() {
            sim.set_reactions_enabled(enabled);
        }
        if e.button("Default").clicked() {
            sim.default_reaction_table();
        }
        if e.button("Clear").clicked() {
            sim.resize_reaction_table();
        }
    });

    // Safety guard warning for GPU reactions
    if ui.use_gpu_physics && sim.params.reactions_enabled && sim.particles.len() > 10_000 {
        e.separator();
        e.colored_label(
            egui::Color32::RED,
            "⚠️ GPU reactions disabled above 10k particles until grid reaction pass exists."
        );
        e.colored_label(
            egui::Color32::YELLOW,
            format!("Current: {} particles (threshold: 10,000)", sim.particles.len())
        );
        e.separator();
    }

    // Copy/Paste and Presets for reaction matrix
    e.horizontal(|e| {
        if e.button("Copy").clicked() {
            let mut matrix = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    matrix.push(sim.rx(i, j));
                }
            }
            ui.reaction_clipboard = Some(matrix);
        }
        if e.button("Paste").clicked() {
            if let Some(ref matrix) = ui.reaction_clipboard {
                if matrix.len() == n * n {
                    sim.edit_reaction_table(|reaction_table, n| {
                        for i in 0..n {
                            for j in 0..n {
                                reaction_table[i * n + j] = matrix[i * n + j];
                            }
                        }
                    });
                }
            }
        }
        if e.button("Random").clicked() {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            sim.edit_reaction_table(|reaction_table, n| {
                for i in 0..n {
                    for j in 0..n {
                        reaction_table[i * n + j] = rng.gen_range(-1..(n as i32));
                    }
                }
            });
        }
        e.checkbox(&mut ui.show_matrix_stats, "Stats");
    });

    // Reaction presets
    e.horizontal(|e| {
        egui::ComboBox::from_label("Reaction Presets")
            .selected_text("Select preset...")
            .show_ui(e, |e| {
                if ui.reaction_presets.is_empty() {
                    ui.reaction_presets = vec![
                        "Rock-Paper-Scissors".to_string(),
                        "Predator-Prey".to_string(),
                        "Chain Reaction".to_string(),
                        "Mutual Transformation".to_string(),
                        "Stable Ecosystem".to_string(),
                    ];
                }
                for preset in &ui.reaction_presets {
                    if e.selectable_label(false, preset).clicked() {
                        apply_reaction_preset(sim, preset);
                    }
                }
            });
    });

    e.horizontal(|e| {
        let mix_radius_max = if ui.cap_to_bounds {
            sim.params.bounds
        } else {
            20.0
        };
        let mut mix_radius = sim.params.mix_radius;
        if e.add(Slider::new(&mut mix_radius, 0.01..=mix_radius_max).text("mix radius"))
            .changed()
        {
            sim.set_mix_radius(mix_radius);
        }
        let mut prob = sim.params.reaction_probability;
        if e.add(Slider::new(&mut prob, 0.01..=1.0).text("probability"))
            .changed()
        {
            sim.set_reaction_probability(prob);
        }
    });

    let mut preserve = sim.params.preserve_particle_count;
    if e.checkbox(&mut preserve, "Preserve particle count")
        .changed()
    {
        sim.set_preserve_particle_count(preserve);
    }
    e.separator();

    // Add scroll area so large matrices don't break window layout
    if n > 16 {
        e.colored_label(
            Color32::YELLOW,
            format!("Matrix hidden: {}x{} is too expensive for live UI. Reduce types or use presets/tools.", n, n)
        );
    } else {
        egui::ScrollArea::both().show(e, |e| {
            Grid::new("reaction_matrix")
                .spacing([2.0, 2.0])
                .show(e, |e| {
                    e.label("");
                    for col in 0..n {
                        e.label(RichText::new(format!("T{}", col)).color(type_color_egui(col)));
                    }
                    e.end_row();

                    for row in 0..n {
                        e.label(RichText::new(format!("T{}", row)).color(type_color_egui(row)));

                        for col in 0..n {
                            let val = sim.rx(row, col);

                            let cell_color = if val >= 0 {
                                let result_type = val as usize;
                                type_color_egui(result_type)
                            } else {
                                Color32::from_rgb(60, 60, 60) // Dark gray for no reaction
                            };

                            let display_text = if val >= 0 {
                                format!("T{}", val)
                            } else {
                                "—".to_string()
                            };

                            let resp = e.add(
                                egui::Button::new(RichText::new(display_text).size(11.0))
                                    .fill(cell_color)
                                    .min_size(egui::vec2(40.0, 28.0)),
                            );

                            // LMB: cycle through reaction results
                            if resp.clicked() {
                                let current = sim.rx(row, col);
                                let next = if current < 0 {
                                    0
                                } else {
                                    (current + 1) % (n as i32)
                                };
                                sim.set_reaction(row, col, next);
                            }

                            // RMB: set to -1 (no reaction)
                            if resp.secondary_clicked() {
                                sim.set_reaction(row, col, -1);
                            }

                            // Scroll fine-tune
                            if resp.hovered() {
                                let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                                if scroll != 0.0 {
                                    let current = sim.rx(row, col);
                                    let new_val = (current + scroll.signum() as i32)
                                        .clamp(-1, (n - 1) as i32);
                                    sim.set_reaction(row, col, new_val);
                                }
                            }
                        }
                        e.end_row();
                    }
                });
        });
    }

    e.separator();
    e.horizontal(|e| {
        e.colored_label(Color32::from_rgb(60, 60, 60), "■ no reaction");
        e.label("= particle transforms to shown type");
    });
}

fn draw_trace_matrix_content(
    ctx: &Context,
    sim: &mut SimState,
    ui: &mut UiState,
    e: &mut egui::Ui,
) {
    draw_trace_matrix_ui(e, ui, sim);
}

fn apply_reaction_preset(sim: &mut SimState, preset: &str) {
    let n = sim.params.type_count;
    match preset {
        "Rock-Paper-Scissors" => {
            sim.edit_reaction_table(|reaction_table, n| {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            reaction_table[i * n + j] = -1; // No self-reaction
                        } else {
                            // Cyclical reactions: 0->1->2->...->0
                            reaction_table[i * n + j] = ((j + n - i - 1) % n) as i32;
                        }
                    }
                }
            });
        }
        "Predator-Prey" => {
            sim.edit_reaction_table(|reaction_table, n| {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            reaction_table[i * n + j] = -1;
                        } else if i % 2 == 0 {
                            reaction_table[i * n + j] = ((i + 1) % n) as i32; // Even types hunt odd types
                        } else {
                            reaction_table[i * n + j] = -1; // Odd types don't hunt
                        }
                    }
                }
            });
        }
        "Chain Reaction" => {
            sim.edit_reaction_table(|reaction_table, n| {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            reaction_table[i * n + j] = -1;
                        } else {
                            reaction_table[i * n + j] = ((i + 1) % n) as i32; // Everything transforms to next type
                        }
                    }
                }
            });
        }
        "Mutual Transformation" => {
            sim.edit_reaction_table(|reaction_table, n| {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            reaction_table[i * n + j] = -1;
                        } else {
                            reaction_table[i * n + j] = ((i + j) % n) as i32; // Sum transforms
                        }
                    }
                }
            });
        }
        "Stable Ecosystem" => {
            sim.edit_reaction_table(|reaction_table, n| {
                for i in 0..n {
                    for j in 0..n {
                        if i == j {
                            reaction_table[i * n + j] = -1;
                        } else if (i + j) % 2 == 0 {
                            reaction_table[i * n + j] = i as i32; // Even pairs stay same
                        } else {
                            reaction_table[i * n + j] = j as i32; // Odd pairs transform to target
                        }
                    }
                }
            });
        }
        _ => {}
    }
}

fn apply_trace_preset(sim: &mut SimState, preset: &str, max_trace: u32) {
    let n = sim.params.type_count;
    match preset {
        "Short Trails" => {
            for i in 0..n {
                for j in 0..n {
                    sim.trace_len_matrix[i * n + j] = (max_trace / 4).max(1);
                }
            }
        }
        "Long Trails" => {
            for i in 0..n {
                for j in 0..n {
                    sim.trace_len_matrix[i * n + j] = max_trace;
                }
            }
        }
        "Diagonal Pattern" => {
            for i in 0..n {
                for j in 0..n {
                    let dist = (i as f32 - j as f32).abs() / n as f32;
                    sim.trace_len_matrix[i * n + j] = ((1.0 - dist) * max_trace as f32) as u32;
                }
            }
        }
        "Cross Pattern" => {
            for i in 0..n {
                for j in 0..n {
                    let is_cross = i == j || (i + j == n - 1);
                    sim.trace_len_matrix[i * n + j] =
                        if is_cross { max_trace } else { max_trace / 2 };
                }
            }
        }
        "Random Burst" => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for i in 0..n {
                for j in 0..n {
                    sim.trace_len_matrix[i * n + j] = if rng.gen_bool(0.3) { max_trace } else { 0 };
                }
            }
        }
        _ => {}
    }
    sim.trace_len_matrix_dirty = true;
}

// ── Creature Book panel ───────────────────────────────────────────────────────
fn draw_book(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    egui::Window::new("📖 Creature Book")
        .default_pos([12.0, 500.0])
        .default_width(300.0)
        .show(ctx, |e| {
            e.label(format!("{} creature(s) saved", sim.book.prefabs.len()));
            e.separator();

            e.label("Save new creature:");
            e.horizontal(|e| {
                e.text_edit_singleline(&mut ui.new_prefab_name);
                if e.button("💾 Save").clicked() && !ui.new_prefab_name.is_empty() {
                    save_selection_as_prefab(sim, ui, ui.new_prefab_name.clone());
                    ui.new_prefab_name.clear();
                }
            });
            e.separator();

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
                        let center = Vec3::splat(sim.params.bounds * 0.5);
                        let instance_id = sim.allocate_prefab_instance_id();
                        sim.spawn_prefab(i, center, instance_id);
                    }
                }
            }
        });
}

fn save_selection_as_prefab(sim: &mut SimState, ui: &UiState, name: String) {
    use crate::sim::book::{Prefab, PrefabParticle};

    if ui.selected_indices.is_empty() {
        log::warn!("No particles selected to save as creature");
        return;
    }

    let selected: Vec<_> = ui
        .selected_indices
        .iter()
        .filter_map(|&idx| sim.particles.get(idx))
        .collect();

    if selected.is_empty() {
        log::warn!("Selected particles not found in simulation");
        return;
    }

    let com: Vec3 = selected
        .iter()
        .map(|p| Vec3::from(p.position))
        .fold(Vec3::ZERO, |a, b| a + b)
        / selected.len() as f32;

    let mut prefab = Prefab::new(name, sim.force_matrix.clone(), sim.params.type_count);
    prefab.particle_count = selected.len();

    for p in selected {
        prefab.particles.push(PrefabParticle {
            relative_position: (Vec3::from(p.position) - com).into(),
            kind: p.kind,
        });
    }

    sim.book.add_prefab(prefab);
}

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
