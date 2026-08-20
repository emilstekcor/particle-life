pub mod audio_ui;
pub mod spawn_ui;

use crate::audio::AudioMod;
use crate::sim::{CpuStepMode, SimState};
use egui::{Color32, Context, Grid, RichText, Slider};
use glam::Vec3;

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
    pub viewport: [u32; 2],
    pub view_matrix: glam::Mat4,
    pub slice_center: f32,
    pub slice_thickness: f32,
    pub move_start_mouse: Option<egui::Pos2>,
    pub move_start_positions: Vec<[f32; 3]>,
    pub pending_assign_type: u32,

    // ── Rule matrix hold-click state ──────────────────────────────────────────

    // ── Trail/trace state ───────────────────────────────────────────────────────
    pub trace_len: u32,
    pub trace_fade_alpha: f32,
    pub trace_render_mode: TraceRenderMode,
    pub trace_type_filter: i32,
    pub trace_trigger_only: bool,
    pub debug_trails: bool, // Enable trail debug output

    // Trace Matrix State
    pub trace_paint_value: u32,
    pub trace_symmetry_lock: bool,
    pub trace_hovered_cell: Option<(usize, usize)>,
    pub trace_brush: TraceBrush,
    pub trace_ui_edit_only: bool,

    // ── Matrix clipboard and presets ───────────────────────────────────────────
    pub force_clipboard: Option<Vec<f32>>,
    pub reaction_clipboard: Option<Vec<i32>>,
    pub trace_clipboard: Option<Vec<u32>>,
    pub show_matrix_stats: bool,
    pub force_presets: Vec<String>,
    pub reaction_presets: Vec<String>,
    pub active_matrix_tab: ActiveMatrixTab,
    
    // Profile fields
    pub save_profile_now: bool,
    pub auto_save_profiles: bool,
    pub auto_save_interval: usize,

    // Request renderer to zero the GPU selection flag buffer (new brush
    // stroke, Clear button, or after deleting particles shifts indices).
    pub clear_selection_requested: bool,

    // ── UX state ────────────────────────────────────────────────────────────
    pub matrix_cell_size: f32,               // shared cell size for all matrix tabs
    pub rules_symmetry: bool,                // mirror edits across the force-matrix diagonal
    pub reaction_paint: i32,                 // palette selection for the reaction tab (-1 = no reaction)
    pub matrix_hovered_cell: Option<(usize, usize)>, // last frame's hovered cell (row/col cross-highlight)
    pub trace_last_painted: Option<(usize, usize)>,  // stroke tracking so Add/Mult brushes fire once per cell
    pub status: Option<(String, f32)>,       // transient toast: (message, seconds left)
    pub styled: bool,                        // one-shot egui style application
    pub strobe: bool,                        // run 2 sim steps per rendered frame (freezes period-2 objects)
    pub last_used_grid: Option<bool>,        // detect Auto mode silently switching step modes

    // ── Audio-driven matrix modulation ────────────────────────────────
    pub audio: AudioMod,

    // ── Spawn composition (pie chart) ─────────────────────────────────
    /// Total pool of particles the chart draws from.
    pub particle_pool: usize,
    /// Per-type counts. Resized to type_count by spawn_ui::sync_mix.
    pub type_mix: Vec<usize>,
    /// Locked types are never auto-adjusted by even-split / steal.
    pub mix_locked: Vec<bool>,
    /// Which wedge has the inline count editor open.
    pub mix_editing: Option<usize>,
    pub mix_edit_buf: String,
    /// Wedge currently being dragged.
    pub mix_drag: Option<usize>,
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
    Audio,
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
                brush_data: [0.0, 0.0, 40.0, 0.0],
                ..Default::default()
            },

            drag_mode: DragMode::None,
            viewport: [1, 1],
            view_matrix: glam::Mat4::IDENTITY,
            slice_center: 0.0,
            slice_thickness: 0.2,
            move_start_mouse: None,
            move_start_positions: Vec::new(),
            pending_assign_type: 0,


            trace_len: 16,
            trace_fade_alpha: 0.98,
            trace_render_mode: TraceRenderMode::Off,
            trace_type_filter: -1,
            trace_trigger_only: false,
            debug_trails: false, // Debug output off by default

            // Trace Matrix State
            trace_paint_value: 30,
            trace_symmetry_lock: false,
            trace_hovered_cell: None,
            trace_brush: TraceBrush::Set,
            trace_ui_edit_only: false,

            // Matrix clipboard and presets
            force_clipboard: None,
            reaction_clipboard: None,
            trace_clipboard: None,
            show_matrix_stats: false,
            force_presets: Vec::new(),
            reaction_presets: Vec::new(),
            active_matrix_tab: ActiveMatrixTab::Rules,
            
            // Profile fields
            save_profile_now: false,
            auto_save_profiles: false,
            auto_save_interval: 500,

            clear_selection_requested: false,

            matrix_cell_size: 32.0,
            rules_symmetry: false,
            reaction_paint: 0,
            matrix_hovered_cell: None,
            trace_last_painted: None,
            status: None,
            styled: false,
            strobe: false,
            last_used_grid: None,

            audio: AudioMod::new(),

            particle_pool: 10_000,
            type_mix: Vec::new(),
            mix_locked: Vec::new(),
            mix_editing: None,
            mix_edit_buf: String::new(),
            mix_drag: None,
        }
    }

    /// Show a small transient toast in the corner (copy/paste feedback, etc).
    pub fn flash(&mut self, msg: impl Into<String>) {
        self.status = Some((msg.into(), 2.0));
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

}

// ── Main UI entry point ───────────────────────────────────────────────────────
pub fn draw_ui(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    if !ui.styled {
        apply_style(ctx);
        ui.styled = true;
    }
    // Audio modulation runs before any panel draws so meters and matrix cells
    // in this frame show the values that were actually uploaded.
    let dt = ctx.input(|i| i.stable_dt).min(0.1);
    ui.audio.tick(sim, dt);
    if ui.audio.playing {
        ctx.request_repaint();
    }

    // Keep the spawn chart in step with type_count, and seed it from the live
    // population on the first frame (SimState::new spawns 512 before any UI runs).
    if ui.type_mix.is_empty() && !sim.particles.is_empty() {
        crate::ui::spawn_ui::adopt_current(sim, ui);
    }
    crate::ui::spawn_ui::sync_mix(sim, ui);

    draw_controls(ctx, sim, ui);
    if ui.show_matrix_editor {
        draw_tabbed_matrices(ctx, sim, ui);
    }
    if ui.show_book {
        draw_book(ctx, sim, ui);
    }
    handle_input(ctx, sim, ui);
    draw_status_flash(ctx, ui);
}

/// One-shot egui style: bigger click targets, consistent rounding, filled sliders.
fn apply_style(ctx: &Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.slider_width = 150.0;
    style.spacing.interact_size.y = 22.0;
    let round = egui::Rounding::same(5.0);
    style.visuals.widgets.inactive.rounding = round;
    style.visuals.widgets.hovered.rounding = round;
    style.visuals.widgets.active.rounding = round;
    style.visuals.widgets.open.rounding = round;
    style.visuals.widgets.noninteractive.rounding = round;
    style.visuals.widgets.hovered.expansion = 1.0;
    style.visuals.window_rounding = egui::Rounding::same(8.0);
    style.visuals.slider_trailing_fill = true;
    ctx.set_style(style);
}

/// Transient toast in the bottom-left corner ("Copied", "Profile saved", ...).
fn draw_status_flash(ctx: &Context, ui: &mut UiState) {
    if let Some((msg, t)) = &mut ui.status {
        *t -= ctx.input(|i| i.stable_dt).min(0.1);
        if *t <= 0.0 {
            ui.status = None;
            return;
        }
        let msg = msg.clone();
        egui::Area::new(egui::Id::new("status_flash"))
            .anchor(egui::Align2::LEFT_BOTTOM, [12.0, -12.0])
            .interactable(false)
            .show(ctx, |e| {
                egui::Frame::popup(&ctx.style()).show(e, |e| {
                    e.label(RichText::new(msg).strong());
                });
            });
        // Keep the timer ticking even when the sim is paused/idle.
        ctx.request_repaint();
    }
}

// ── Matrix cell widgets ─────────────────────────────────────────────────────
fn cell_dims(size: f32) -> egui::Vec2 {
    egui::vec2(size, (size * 0.78).max(16.0))
}

/// Is the pointer over this cell right now?
///
/// `Response::hovered()` goes false the moment any widget captures the pointer
/// press — including a ScrollArea's drag-to-scroll — so it can't be used to
/// drive drag-painting. Hit-testing the rect directly survives the press.
fn cell_hit(e: &egui::Ui, resp: &egui::Response) -> bool {
    e.input(|i| i.pointer.interact_pos()).map_or(false, |p| {
        resp.rect.contains(p) && e.clip_rect().contains(p)
    })
}

/// A painted matrix cell. Cheap (no Button layout), hover-brightened, with
/// optional centered monospace text in a contrast color.
fn matrix_cell(
    e: &mut egui::Ui,
    size: f32,
    fill: Color32,
    text: Option<String>,
    sense: egui::Sense,
    selected: bool,
) -> egui::Response {
    let (rect, resp) = e.allocate_exact_size(cell_dims(size), sense);
    if e.is_rect_visible(rect) {
        let pointer_over = e
            .input(|i| i.pointer.interact_pos())
            .map_or(false, |p| rect.contains(p) && e.clip_rect().contains(p));
        let active = resp.hovered() || resp.dragged() || pointer_over;
        let fill = if active {
            Color32::from_rgb(
                fill.r().saturating_add(26),
                fill.g().saturating_add(26),
                fill.b().saturating_add(26),
            )
        } else {
            fill
        };
        let painter = e.painter();
        painter.rect_filled(rect, 3.0, fill);
        if selected {
            painter.rect_stroke(rect, 3.0, egui::Stroke::new(2.0, Color32::WHITE));
        } else if active {
            painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, Color32::from_gray(220)));
        }
        if let Some(t) = text {
            let lum = fill.r() as u32 + fill.g() as u32 + fill.b() as u32;
            let text_color = if lum > 400 {
                Color32::from_gray(15)
            } else {
                Color32::from_gray(235)
            };
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                t,
                egui::FontId::monospace((size * 0.32).clamp(9.0, 12.0)),
                text_color,
            );
        }
    }
    resp
}

/// Row/column header: a colored dot for the particle type, with a subtle
/// backing when its row/column is hovered.
fn type_header(e: &mut egui::Ui, idx: usize, size: f32, highlight: bool) {
    let (rect, resp) = e.allocate_exact_size(cell_dims(size), egui::Sense::hover());
    if e.is_rect_visible(rect) {
        let painter = e.painter();
        if highlight {
            painter.rect_filled(rect, 3.0, Color32::from_gray(70));
        }
        painter.circle_filled(
            rect.center(),
            (size * 0.20).clamp(4.0, 8.0),
            type_color_egui(idx),
        );
    }
    resp.on_hover_text(format!("Type {idx}"));
}

/// Auto-shrink cells for big matrices so 32×32 still fits on screen.
fn effective_cell_size(user: f32, n: usize) -> f32 {
    if n > 16 {
        user.min(22.0)
    } else {
        user
    }
}

/// Read and consume vertical scroll while hovering a matrix cell, so
/// fine-tuning a value doesn't also scroll the surrounding ScrollArea.
fn take_scroll(ctx: &Context) -> f32 {
    ctx.input_mut(|i| {
        let s = i.smooth_scroll_delta.y;
        i.smooth_scroll_delta.y = 0.0;
        s
    })
}

// ── Tabbed Matrix Editor ───────────────────────────────────────────────────────
fn draw_tabbed_matrices(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    egui::Window::new("Matrix Editor")
        .default_pos([350.0, 12.0])
        .resizable(true)
        .default_width(600.0)
        .show(ctx, |e| {
            // Tab selection + shared cell-size control
            e.horizontal(|e| {
                e.selectable_value(&mut ui.active_matrix_tab, ActiveMatrixTab::Rules, "Rules");
                e.selectable_value(
                    &mut ui.active_matrix_tab,
                    ActiveMatrixTab::Reactions,
                    "Reactions",
                );
                e.selectable_value(&mut ui.active_matrix_tab, ActiveMatrixTab::Traces, "Traces");
                e.selectable_value(&mut ui.active_matrix_tab, ActiveMatrixTab::Audio, "Audio");
                e.with_layout(egui::Layout::right_to_left(egui::Align::Center), |e| {
                    e.add(
                        egui::DragValue::new(&mut ui.matrix_cell_size)
                            .speed(0.5)
                            .clamp_range(18.0..=48.0)
                            .suffix(" px"),
                    )
                    .on_hover_text("Matrix cell size (drag)");
                    e.label("cells:");
                });
            });
            e.separator();

            // Show content based on active tab
            match ui.active_matrix_tab {
                ActiveMatrixTab::Rules => draw_rule_matrix_content(ctx, sim, ui, e),
                ActiveMatrixTab::Reactions => draw_reaction_matrix_content(ctx, sim, ui, e),
                ActiveMatrixTab::Traces => draw_trace_matrix_content(ctx, sim, ui, e),
                ActiveMatrixTab::Audio => {
                    crate::ui::audio_ui::draw_audio_matrix_content(ctx, sim, ui, e)
                }
            }
        });
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

        ui.add(
            egui::Slider::new(&mut state.trace_len, 1..=crate::renderer::compute::MAX_TRAIL)
                .text("history"),
        );
        ui.add(egui::Slider::new(&mut state.trace_fade_alpha, 0.0..=1.0).text("fade"));
        ui.checkbox(&mut state.trace_ui_edit_only, "edit only")
            .on_hover_text("Edit the trace matrix without rendering trails");
    });

    ui.separator();
    ui.label(
        RichText::new("pick a brush · click/drag cells to paint · right-click erase").weak(),
    );

    ui.horizontal(|ui| {
        ui.label("Brush:");
        egui::ComboBox::from_id_source("trace_brush")
            .selected_text(format!("{:?}", state.trace_brush))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut state.trace_brush, TraceBrush::Set, "Set");
                ui.selectable_value(&mut state.trace_brush, TraceBrush::Add, "Add");
                ui.selectable_value(&mut state.trace_brush, TraceBrush::Subtract, "Subtract");
                ui.selectable_value(&mut state.trace_brush, TraceBrush::Multiply, "Multiply");
                ui.selectable_value(&mut state.trace_brush, TraceBrush::Erase, "Erase");
            });

        ui.label("lifetime:");
        ui.add(
            egui::DragValue::new(&mut state.trace_paint_value)
                .speed(1)
                .clamp_range(0..=600),
        )
        .on_hover_text("Value painted into cells (steps)");

        ui.checkbox(&mut state.trace_symmetry_lock, "mirror")
            .on_hover_text("Edits apply to both (i,j) and (j,i)");
    });

    ui.horizontal_wrapped(|ui| {
        if ui.button("Clear all").clicked() {
            for v in &mut sim.trace_len_matrix {
                *v = 0;
            }
            sim.trace_len_matrix_dirty = true;
            state.flash("Trace matrix cleared");
        }
        if ui.button("Fill all").clicked() {
            for v in &mut sim.trace_len_matrix {
                *v = state.trace_paint_value;
            }
            sim.trace_len_matrix_dirty = true;
            state.flash("Trace matrix filled");
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
        if ui.button("Copy").clicked() {
            state.trace_clipboard = Some(sim.trace_len_matrix.clone());
            state.flash("Trace matrix copied");
        }
        if ui.button("Paste").clicked() {
            if let Some(ref matrix) = state.trace_clipboard {
                if matrix.len() == n * n {
                    sim.trace_len_matrix.copy_from_slice(matrix);
                    sim.trace_len_matrix_dirty = true;
                    state.flash("Trace matrix pasted");
                } else {
                    state.flash("Clipboard is a different size");
                }
            }
        }
        egui::ComboBox::from_id_source("trace_presets")
            .selected_text("Presets...")
            .show_ui(ui, |ui| {
                for preset in [
                    "Short Trails",
                    "Long Trails",
                    "Diagonal Pattern",
                    "Cross Pattern",
                    "Random Burst",
                ] {
                    if ui.selectable_label(false, preset).clicked() {
                        apply_trace_preset(sim, preset, state.trace_paint_value.max(1));
                        state.flash(format!("Preset: {preset}"));
                    }
                }
            });
        if ui.button("From reactions").clicked() {
            for r in 0..n {
                for c in 0..n {
                    let reaction_idx = r * n + c;
                    let trace_idx = r * n + c;
                    if trace_idx < sim.trace_len_matrix.len()
                        && reaction_idx < sim.reaction_table.len()
                    {
                        sim.trace_len_matrix[trace_idx] =
                            if sim.reaction_table[reaction_idx] > 0 {
                                state.trace_paint_value
                            } else {
                                0
                            };
                    }
                }
            }
            sim.trace_len_matrix_dirty = true;
        }
    });

    let active = sim.trace_len_matrix.iter().filter(|&&v| v > 0).count();
    ui.label(RichText::new(format!("Active trace cells: {}/{}", active, n * n)).weak());

    ui.separator();

    if sim.trace_len_matrix.len() != n * n {
        sim.trace_len_matrix.resize(n * n, 0);
        sim.trace_len_matrix_dirty = true;
    }

    // Find max value for heatmap normalization
    let max_value = sim.trace_len_matrix.iter().copied().max().unwrap_or(1).max(1);

    let hovered_last = state.trace_hovered_cell;
    state.trace_hovered_cell = None;
    let cell = effective_cell_size(state.matrix_cell_size, n);
    let show_text = cell >= 28.0;
    let (pdown, sdown) = ui.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));
    if !pdown {
        // Stroke ended: Add/Subtract/Multiply brushes may fire again per cell
        state.trace_last_painted = None;
    }

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .drag_to_scroll(false) // otherwise the ScrollArea eats the paint drag
        .max_height(420.0)
        .show(ui, |ui| {
            egui::Grid::new("trace_matrix_grid")
                .spacing([2.0, 2.0])
                .show(ui, |ui| {
                    ui.label("");
                    for c in 0..n {
                        type_header(ui, c, cell, hovered_last.map_or(false, |(_, cc)| cc == c));
                    }
                    ui.end_row();

                    for r in 0..n {
                        type_header(ui, r, cell, hovered_last.map_or(false, |(rr, _)| rr == r));

                        for c in 0..n {
                            let idx = r * n + c;
                            let value = sim.trace_len_matrix[idx];

                            // Heatmap: dark → amber
                            let color = if value == 0 {
                                Color32::from_rgb(38, 38, 42)
                            } else {
                                let t = (value as f32 / max_value as f32).clamp(0.0, 1.0);
                                let lerp =
                                    |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t) as u8;
                                Color32::from_rgb(lerp(70, 255), lerp(72, 178), lerp(110, 44))
                            };

                            // Sense::hover so painting sweeps across cells while dragging
                            let resp = matrix_cell(
                                ui,
                                cell,
                                color,
                                show_text.then(|| {
                                    if value == 0 {
                                        "·".to_string()
                                    } else {
                                        value.to_string()
                                    }
                                }),
                                egui::Sense::hover(),
                                false,
                            );

                            // Rect hit-test, not resp.hovered(): hover is
                            // suppressed for the whole duration of a press.
                            if cell_hit(ui, &resp) {
                                state.trace_hovered_cell = Some((r, c));

                                if pdown {
                                    // Fire once per cell per stroke so Add/Multiply
                                    // don't run away at 60 fps
                                    if state.trace_last_painted != Some((r, c)) {
                                        state.trace_last_painted = Some((r, c));
                                        let paint = state.trace_paint_value;
                                        let new_value = match state.trace_brush {
                                            TraceBrush::Set => paint,
                                            TraceBrush::Add => value.saturating_add(paint),
                                            TraceBrush::Subtract => value.saturating_sub(paint),
                                            TraceBrush::Multiply => value.saturating_mul(paint),
                                            TraceBrush::Erase => 0,
                                        };
                                        if new_value != value {
                                            set_trace_cell(
                                                sim,
                                                r,
                                                c,
                                                new_value,
                                                state.trace_symmetry_lock,
                                            );
                                        }
                                    }
                                } else if sdown && value != 0 {
                                    set_trace_cell(sim, r, c, 0, state.trace_symmetry_lock);
                                }
                            }

                            resp.on_hover_text(format!(
                                "T{r} + T{c}: {value} steps\npaint = LMB drag · erase = RMB"
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

    // ── Keyboard shortcuts (skipped while typing in a text field) ─────────────
    if !ctx.wants_keyboard_input() {
        ctx.input(|i| {
            use egui::Key;
            if i.key_pressed(Key::Space) {
                ui.paused = !ui.paused;
            }
            if i.key_pressed(Key::N) {
                ui.step_once = true;
            }
            if i.key_pressed(Key::M) {
                ui.show_matrix_editor = !ui.show_matrix_editor;
            }
            if i.key_pressed(Key::B) {
                ui.show_book = !ui.show_book;
            }
            if i.key_pressed(Key::Num1) {
                ui.selection_mode = SelectionMode::Rect;
            }
            if i.key_pressed(Key::Num2) {
                ui.selection_mode = SelectionMode::Brush;
            }
            if i.key_pressed(Key::Num3) {
                ui.selection_mode = SelectionMode::Slice;
            }
        });
    }

    // Toggle trail with T key - cycle through functional modes (skip Simple)
    if !ctx.wants_keyboard_input() && ctx.input(|i| i.key_pressed(egui::Key::T)) {
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

    // Idle: deactivate the selection tool. Mode 0 preserves existing GPU
    // selection flags, so a finished selection keeps its highlight.
    if ui.drag_mode == DragMode::None && !pointer.any_down() {
        ui.gpu_selection_params.mode_flags[0] = 0;
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
                    // Brush accumulates along the stroke, so a new stroke
                    // starts from an empty selection.
                    ui.clear_selection_requested = true;
                    ui.gpu_selection_params.mode_flags[0] = 2; // Brush mode
                    ui.gpu_selection_params.brush_data = [pos.x, pos.y, ui.brush_radius, 0.0];
                }
                SelectionMode::Slice => {
                    ui.gpu_selection_params.mode_flags[0] = 3; // Slice mode
                    ui.gpu_selection_params.brush_data =
                        [0.0, 0.0, ui.slice_thickness, ui.slice_center];
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
                    ui.gpu_selection_params.brush_data =
                        [0.0, 0.0, ui.slice_thickness, ui.slice_center];
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
        .resizable(true)
        .show(ctx, |e| {
            // The panel outgrew most screens once Spawn gained the chart, so the
            // whole body scrolls. auto_shrink keeps it from collapsing to nothing
            // when a section is folded away.
            egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .id_source("controls_scroll")
            .show(e, |e| {
            e.label(format!("Particles: {}", sim.particles.len()));
            e.label(format!("Types:     {}", sim.params.type_count));
            e.label(format!("Steps:     {}", sim.step_count));
            e.separator();

            e.horizontal(|e| {
                let lbl = if ui.paused { "▶ Resume" } else { "⏸ Pause" };
                if e.add_sized([140.0, 28.0], egui::Button::new(RichText::new(lbl).size(15.0)))
                    .on_hover_text("Space")
                    .clicked()
                {
                    ui.paused = !ui.paused;
                }
                if e.add_sized([70.0, 28.0], egui::Button::new("⏭ Step"))
                    .on_hover_text("N — while strobing, one Step flips the visible phase")
                    .clicked()
                {
                    ui.step_once = true;
                }
            });
            e.checkbox(&mut ui.strobe, "Strobe ×2").on_hover_text(
                "Run two sim steps per rendered frame so period-2 oscillating\n\
                 objects appear frozen instead of flickering.\n\
                 ⏭ Step (while paused) advances one step to view the other phase.",
            );

            // Auto step mode can silently swap the interaction model at the
            // particle-count threshold — announce it, because a knife-edge
            // object dissolving for no visible reason is maddening.
            if !ui.use_gpu_physics {
                let used = sim.last_step_used_grid;
                if let Some(prev) = ui.last_used_grid {
                    if prev != used && matches!(sim.params.cpu_step_mode, CpuStepMode::Auto) {
                        ui.flash(if used {
                            "Auto: CPU switched to GridExact"
                        } else {
                            "Auto: CPU switched to Naive"
                        });
                    }
                }
                ui.last_used_grid = Some(used);
            } else {
                ui.last_used_grid = None;
            }
            e.separator();

            e.checkbox(&mut ui.use_gpu_physics, "Use GPU Physics");
            e.separator();

            let mut changed = false;
            egui::CollapsingHeader::new("Physics")
                .default_open(true)
                .show(e, |e| {

                let mut type_count = sim.params.type_count;
                if e.add(
                    Slider::new(&mut type_count, 1..=crate::renderer::compute::MAX_TYPES)
                        .text("types"),
                )
                .changed()
                {
                    if type_count != sim.params.type_count {
                        sim.set_type_count(type_count);
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
                });

            e.checkbox(&mut ui.show_matrix_editor, "Matrix Editor")
                .on_hover_text("M");

            egui::CollapsingHeader::new("CPU stepping & timing")
                .default_open(false)
                .show(e, |e| {
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
                });

            if changed {
                sim.params_dirty = true;
            }
            e.separator();

            egui::CollapsingHeader::new("Spawn")
                .default_open(true)
                .show(e, |e| {
                crate::ui::spawn_ui::draw_spawn_panel(e, sim, ui);
                e.horizontal(|e| {
                    if e.button("🎲 Rules").clicked() {
                        sim.randomize_rules();
                        ui.flash("Rules randomized");
                    }
                });
                });
            e.separator();

            egui::CollapsingHeader::new("Camera")
                .default_open(false)
                .show(e, |e| {
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
                });
            e.separator();

            egui::CollapsingHeader::new("Selection")
                .default_open(false)
                .show(e, |e| {
                e.horizontal(|e| {
                    if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Rect), "Rect")
                        .on_hover_text("1")
                        .clicked()
                    {
                        ui.selection_mode = SelectionMode::Rect;
                    }
                    if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Brush), "Brush")
                        .on_hover_text("2")
                        .clicked()
                    {
                        ui.selection_mode = SelectionMode::Brush;
                    }
                    if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Slice), "Slice")
                        .on_hover_text("3")
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
                            ui.clear_selection_requested = true;
                        }
                        if e.button("Delete").clicked() {
                            let count = ui.selected_indices.len();
                            sim.delete_particles(&ui.selected_indices);
                            ui.selected_indices.clear();
                            // Indices shifted; stale GPU flags would highlight the
                            // wrong particles.
                            ui.clear_selection_requested = true;
                            ui.flash(format!("Deleted {count} particles"));
                        }
                        if e.button("Duplicate").clicked() {
                            sim.duplicate_particles(&ui.selected_indices);
                            ui.flash(format!("Duplicated {} particles", ui.selected_indices.len()));
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
                            ui.flash(format!(
                                "Assigned T{} to {} particles",
                                ui.pending_assign_type,
                                ui.selected_indices.len()
                            ));
                        }
                    });
                }
                });
            e.separator();

            egui::CollapsingHeader::new("Profiles")
                .default_open(false)
                .show(e, |e| {
                e.horizontal(|e| {
                    if e.button("💾 Save profile").clicked() {
                        ui.save_profile_now = true;
                    }
                    e.checkbox(&mut ui.auto_save_profiles, "auto-save");
                });
                if ui.auto_save_profiles {
                    e.add(
                        Slider::new(&mut ui.auto_save_interval, 100..=5000)
                            .text("every N steps"),
                    );
                }
                });
            e.separator();

            if e.button("📖 Creature Book").on_hover_text("B").clicked() {
                ui.show_book = !ui.show_book;
            }
            });
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

    e.label(
        RichText::new("drag ↕ adjust · scroll fine · double-click zero · right-click flip sign")
            .weak(),
    );
    e.separator();

    // Enhanced controls row
    e.horizontal(|e| {
        if e.button("Zero all").clicked() {
            for i in 0..n {
                for j in 0..n {
                    sim.set_rule(i, j, 0.0);
                }
            }
            ui.flash("Force matrix zeroed");
        }
        if e.button("Symmetrize").clicked() {
            for i in 0..n {
                for j in (i + 1)..n {
                    let avg = (sim.get_rule(i, j) + sim.get_rule(j, i)) * 0.5;
                    sim.set_rule(i, j, avg);
                    sim.set_rule(j, i, avg);
                }
            }
            ui.flash("Symmetrized");
        }
        if e.button("Invert").clicked() {
            for i in 0..n {
                for j in 0..n {
                    let v = sim.get_rule(i, j);
                    sim.set_rule(i, j, -v);
                }
            }
            ui.flash("Inverted");
        }
        e.checkbox(&mut ui.rules_symmetry, "mirror")
            .on_hover_text("Edits apply to both (i,j) and (j,i)");
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
            ui.flash("Force matrix copied");
        }
        if e.button("Paste").clicked() {
            if let Some(ref matrix) = ui.force_clipboard {
                if matrix.len() == n * n {
                    for i in 0..n {
                        for j in 0..n {
                            sim.set_rule(i, j, matrix[i * n + j]);
                        }
                    }
                    ui.flash("Force matrix pasted");
                } else {
                    ui.flash("Clipboard is a different size");
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
            ui.flash("Force matrix randomized");
        }
        e.checkbox(&mut ui.show_matrix_stats, "Stats");
    });

    // Audio modulation moved to its own tab (ActiveMatrixTab::Audio).

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

    let hovered_last = ui.matrix_hovered_cell;
    ui.matrix_hovered_cell = None;
    let cell = effective_cell_size(ui.matrix_cell_size, n);
    let show_text = cell >= 28.0;

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .drag_to_scroll(false) // cell drags adjust values, not scroll position
        .max_height(560.0)
        .show(e, |e| {
            Grid::new("matrix").spacing([2.0, 2.0]).show(e, |e| {
                e.label("");
                for col in 0..n {
                    type_header(e, col, cell, hovered_last.map_or(false, |(_, c)| c == col));
                }
                e.end_row();

                for row in 0..n {
                    type_header(e, row, cell, hovered_last.map_or(false, |(r, _)| r == row));

                    for col in 0..n {
                        let val = sim.get_rule(row, col);

                        let cell_color = if val > 0.0 {
                            Color32::from_rgb(30, (40.0 + val * 180.0) as u8, 30)
                        } else if val < 0.0 {
                            Color32::from_rgb((40.0 + val.abs() * 180.0) as u8, 30, 30)
                        } else {
                            Color32::from_rgb(40, 40, 40)
                        };

                        let resp = matrix_cell(
                            e,
                            cell,
                            cell_color,
                            show_text.then(|| format!("{:+.2}", val)),
                            egui::Sense::click_and_drag(),
                            false,
                        );

                        let mut new_val = val;
                        let mut changed = false;

                        // Drag vertically to adjust: 200 px covers the full range
                        if resp.dragged() {
                            new_val = (new_val - resp.drag_delta().y * 0.005).clamp(-1.0, 1.0);
                            changed = new_val != val;
                            e.output_mut(|o| o.cursor_icon = egui::CursorIcon::ResizeVertical);
                        }
                        if resp.double_clicked() {
                            new_val = 0.0;
                            changed = true;
                        }
                        if resp.secondary_clicked() {
                            new_val = -val;
                            changed = val != 0.0;
                        }
                        if resp.hovered() {
                            ui.matrix_hovered_cell = Some((row, col));
                            // Consume scroll so fine-tuning doesn't also pan the ScrollArea
                            let scroll = take_scroll(ctx);
                            if scroll != 0.0 {
                                new_val = (new_val + scroll * 0.002).clamp(-1.0, 1.0);
                                changed = true;
                            }
                        }

                        if changed {
                            sim.set_rule(row, col, new_val);
                            if ui.rules_symmetry && row != col {
                                sim.set_rule(col, row, new_val);
                            }
                        }

                        resp.on_hover_text(format!(
                            "T{row} → T{col}: {val:+.3}\ndrag ↕ adjust · scroll fine\ndouble-click zero · right-click flip sign"
                        ));
                    }
                    e.end_row();
                }
            });
        });

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

    e.label(
        RichText::new("pick a paint below · click/drag cells to paint · right-click erase · scroll cycle")
            .weak(),
    );
    e.separator();

    // Reaction controls
    e.horizontal(|e| {
        let mut enabled = sim.params.reactions_enabled;
        if e.checkbox(&mut enabled, "Enable Reactions").changed() {
            sim.set_reactions_enabled(enabled);
        }
        if e.button("Default").clicked() {
            sim.default_reaction_table();
            ui.flash("Default reactions loaded");
        }
        if e.button("Clear").clicked() {
            sim.resize_reaction_table();
            ui.flash("Reactions cleared");
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
            ui.flash("Reaction table copied");
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
                    ui.flash("Reaction table pasted");
                } else {
                    ui.flash("Clipboard is a different size");
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
            ui.flash("Reaction table randomized");
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

    // Palette: choose what LMB paints
    if ui.reaction_paint >= n as i32 {
        ui.reaction_paint = -1;
    }
    e.horizontal_wrapped(|e| {
        e.label("Paint:");
        let resp = matrix_cell(
            e,
            26.0,
            Color32::from_rgb(55, 55, 55),
            Some("—".into()),
            egui::Sense::click(),
            ui.reaction_paint == -1,
        )
        .on_hover_text("Paint: no reaction");
        if resp.clicked() {
            ui.reaction_paint = -1;
        }
        for t in 0..n {
            let resp = matrix_cell(
                e,
                26.0,
                type_color_egui(t),
                None,
                egui::Sense::click(),
                ui.reaction_paint == t as i32,
            )
            .on_hover_text(format!("Paint: T{t}"));
            if resp.clicked() {
                ui.reaction_paint = t as i32;
            }
        }
    });
    e.separator();

    let hovered_last = ui.matrix_hovered_cell;
    ui.matrix_hovered_cell = None;
    let cell = effective_cell_size(ui.matrix_cell_size, n);
    let show_text = cell >= 28.0;
    let (pdown, sdown) = e.input(|i| (i.pointer.primary_down(), i.pointer.secondary_down()));

    egui::ScrollArea::both()
        .auto_shrink([false, false])
        .drag_to_scroll(false) // otherwise the ScrollArea eats the paint drag
        .max_height(520.0)
        .show(e, |e| {
            Grid::new("reaction_matrix")
                .spacing([2.0, 2.0])
                .show(e, |e| {
                    e.label("");
                    for col in 0..n {
                        type_header(e, col, cell, hovered_last.map_or(false, |(_, c)| c == col));
                    }
                    e.end_row();

                    for row in 0..n {
                        type_header(e, row, cell, hovered_last.map_or(false, |(r, _)| r == row));

                        for col in 0..n {
                            let val = sim.rx(row, col);

                            let cell_color = if val >= 0 {
                                type_color_egui(val as usize)
                            } else {
                                Color32::from_rgb(55, 55, 55) // no reaction
                            };
                            let display_text = if val >= 0 {
                                format!("T{}", val)
                            } else {
                                "—".to_string()
                            };

                            // Sense::hover so drag-painting sweeps across cells
                            let resp = matrix_cell(
                                e,
                                cell,
                                cell_color,
                                show_text.then_some(display_text),
                                egui::Sense::hover(),
                                false,
                            );

                            // Rect hit-test, not resp.hovered(): hover is
                            // suppressed for the whole duration of a press.
                            if cell_hit(e, &resp) {
                                ui.matrix_hovered_cell = Some((row, col));

                                if pdown && val != ui.reaction_paint {
                                    sim.set_reaction(row, col, ui.reaction_paint);
                                } else if sdown && val != -1 {
                                    sim.set_reaction(row, col, -1);
                                }

                                let scroll = take_scroll(ctx);
                                if scroll != 0.0 {
                                    let new_val = (val + scroll.signum() as i32)
                                        .clamp(-1, (n - 1) as i32);
                                    sim.set_reaction(row, col, new_val);
                                }
                            }

                            let result = if val >= 0 {
                                format!("T{val}")
                            } else {
                                "no reaction".to_string()
                            };
                            resp.on_hover_text(format!("T{row} + T{col} → {result}"));
                        }
                        e.end_row();
                    }
                });
        });

    e.separator();
    e.horizontal(|e| {
        e.colored_label(Color32::from_rgb(60, 60, 60), "■ no reaction");
        e.label("= particle transforms to shown type");
    });
}

fn draw_trace_matrix_content(
    _ctx: &Context,
    sim: &mut SimState,
    ui: &mut UiState,
    e: &mut egui::Ui,
) {
    draw_trace_matrix_ui(e, ui, sim);
}

fn apply_reaction_preset(sim: &mut SimState, preset: &str) {
    let _n = sim.params.type_count;
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


