use egui::{Context, Slider, Grid, RichText, Color32};
use crate::sim::{SimState, CpuStepMode};
use glam::{Vec3, Vec4Swizzles};

// ── UiState ───────────────────────────────────────────────────────────────────
pub struct UiState {
    pub paused:    bool,
    pub step_once: bool,
    pub use_gpu_physics: bool,
    pub cap_to_bounds: bool,

    // ── Free-fly camera ───────────────────────────────────────────────────────
    pub fly_pos:   Vec3,
    pub fly_yaw:   f32,
    pub fly_pitch: f32,
    pub fly_speed: f32,
    pub camera_mode: bool,
    pub mouse_look_active: bool,

    // camera_dist is now unused since slice_center is user-controlled

    // ── Book panel ────────────────────────────────────────────────────────────
    pub show_book: bool,
    pub new_prefab_name: String,

    // ── Selection ─────────────────────────────────────────────────────────────
    pub selected_prefab: Option<usize>,
    pub selected_indices: Vec<usize>,
    pub highlighted_indices: Vec<usize>,
    pub selection_mode: SelectionMode,
    pub brush_radius: f32,
    pub drag_start: Option<egui::Pos2>,
    pub drag_end:   Option<egui::Pos2>,
    pub is_selecting: bool,
    pub selection_readback_needed: bool,
    pub gpu_selection_params: crate::renderer::compute::SelectionParams,

    // ── Camera projection matrices (written by renderer each frame) ───────────
    pub view_proj:   glam::Mat4,
    pub drag_mode:   DragMode,
    pub hover_index: Option<usize>,
    pub viewport:    [u32; 2],
    pub view_matrix: glam::Mat4,
    pub slice_center:    f32,
    pub slice_thickness: f32,
    pub move_start_mouse:     Option<egui::Pos2>,
    pub move_start_positions: Vec<[f32; 3]>,
    pub pending_assign_type: u32,

    // ── Rule matrix hold-click state ──────────────────────────────────────────
    pub matrix_hold: Option<(usize, usize, f32)>,
    pub matrix_hold_timer: f32,

    // ── Trail/trace state ───────────────────────────────────────────────────────
    pub traces: bool,           // Java-style simple boolean
    pub trace_len: u32,
    pub trace_fade_alpha: f32,
    pub trace_render_mode: TraceRenderMode,
    pub trace_type_filter: i32,
    pub debug_trails: bool,     // Enable trail debug output
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode { Rect, Brush, Slice }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceRenderMode {
    Off,
    Simple,    // Java-style framebuffer accumulation
    Lines,
    Dots,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionOp { Replace, Add, Remove }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragMode { None, Selecting, MovingSelection }

impl UiState {
    pub fn new() -> Self {
        Self {
            paused:    false,
            step_once: false,
            use_gpu_physics: false,
            cap_to_bounds: true,

            fly_pos:   Vec3::new(0.5, 0.5, 2.0),
            fly_yaw:   -std::f32::consts::PI / 2.0, // Point toward -Z (center)
            fly_pitch: 0.0,
            fly_speed: 0.5,
            camera_mode: false,
            mouse_look_active: false,

            // camera_dist removed - slice_center is now user-controlled

            show_book:       false,
            new_prefab_name: String::new(),
            selected_prefab: None,

            selected_indices:    Vec::new(),
            highlighted_indices: Vec::new(),
            selection_mode: SelectionMode::Rect,
            brush_radius:   40.0,
            drag_start: None,
            drag_end:   None,
            is_selecting: false,
            selection_readback_needed: false,
            view_proj:   glam::Mat4::IDENTITY,
            gpu_selection_params: crate::renderer::compute::SelectionParams {
                mode: 0,
                _pad0: [0; 3],
                rect_min: [0.0, 0.0],
                _pad1: [0.0, 0.0],
                rect_max: [0.0, 0.0],
                _pad2: [0.0, 0.0],
                brush_center: [0.0, 0.0],
                _pad3: [0.0, 0.0],
                brush_radius: 40.0,
                slice_depth: 0.0,
                _pad4: [0.0, 0.0],
},

            drag_mode:   DragMode::None,
            hover_index: None,
            viewport:    [1, 1],
            view_matrix: glam::Mat4::IDENTITY,
            slice_center:    0.0,
            slice_thickness: 0.2,
            move_start_mouse:     None,
            move_start_positions: Vec::new(),
            pending_assign_type:  0,

            matrix_hold:       None,
            matrix_hold_timer: 0.0,

            traces: false,          // Java-style: start with traces off
            trace_len: 16,
            trace_fade_alpha: 0.98,
            trace_render_mode: TraceRenderMode::Off,
            trace_type_filter: -1,
            debug_trails: false,   // Debug output off by default
        }
    }

    /// Forward direction in world space from current yaw/pitch
    pub fn fly_forward(&self) -> Vec3 {
        Vec3::new(
            self.fly_yaw.cos() * self.fly_pitch.cos(),
            self.fly_pitch.sin(),
            self.fly_yaw.sin() * self.fly_pitch.cos(),
        ).normalize()
    }

    /// Right vector (perpendicular to forward, in the XZ plane)
    pub fn fly_right(&self) -> Vec3 {
        self.fly_forward().cross(Vec3::Y).normalize()
    }

    #[allow(dead_code)]
    pub fn world_to_screen(&self, pos: Vec3, viewport: [u32; 2]) -> Option<egui::Pos2> {
        let clip = self.view_proj * pos.extend(1.0);
        if clip.w <= 0.0 { return None; }
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
    draw_rule_matrix(ctx, sim, ui);
    draw_reaction_matrix(ctx, sim, ui);
    if ui.show_book {
        draw_book(ctx, sim, ui);
    }
    handle_input(ctx, sim, ui);
}

// ── Selection helpers ─────────────────────────────────────────────────────────
fn apply_selection_op(selected: &mut Vec<usize>, hits: &[usize], op: SelectionOp) {
    match op {
        SelectionOp::Replace => { selected.clear(); selected.extend_from_slice(hits); }
        SelectionOp::Add     => { selected.extend_from_slice(hits); }
        SelectionOp::Remove  => {
            let rm: std::collections::HashSet<_> = hits.iter().copied().collect();
            selected.retain(|i| !rm.contains(i));
        }
    }
    dedup_selection(selected);
}

fn dedup_selection(v: &mut Vec<usize>) { v.sort_unstable(); v.dedup(); }

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
        let fwd   = ui.fly_forward();
        let right = ui.fly_right();
        let up    = Vec3::Y;
        let speed = ui.fly_speed;

        ctx.input(|i| {
            use egui::Key;
            if i.key_down(Key::W) { ui.fly_pos += fwd   * speed * dt; }
            if i.key_down(Key::S) { ui.fly_pos -= fwd   * speed * dt; }
            if i.key_down(Key::D) { ui.fly_pos += right * speed * dt; }
            if i.key_down(Key::A) { ui.fly_pos -= right * speed * dt; }
            if i.key_down(Key::E) { ui.fly_pos += up    * speed * dt; }
            if i.key_down(Key::Q) { ui.fly_pos -= up    * speed * dt; }
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
                ui.fly_yaw   += delta.x * 0.003;
                ui.fly_pitch  = (ui.fly_pitch - delta.y * 0.003)
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

    let pointer   = ctx.input(|i| i.pointer.clone());
    let latest    = pointer.latest_pos();

    // Active move-drag takes full control
    if ui.drag_mode == DragMode::MovingSelection {
        handle_move_drag(pointer, latest, sim, ui);
        return;
    }

    // Hover preview when idle - clear GPU selection when not interacting
    if ui.drag_mode == DragMode::None && !pointer.any_down() {
        ui.gpu_selection_params.mode = 0; // No selection
        ui.hover_index = None;
        ui.highlighted_indices.clear();
    }

    if pointer.primary_pressed() {
        if let Some(pos) = pointer.press_origin() {
            ui.drag_start = Some(pos);
            ui.drag_end   = Some(pos);
            ui.drag_mode  = DragMode::Selecting;
            ui.is_selecting = true;

            // Set GPU selection parameters based on mode
            match ui.selection_mode {
                SelectionMode::Rect => {
                    ui.gpu_selection_params.mode = 1; // Rect mode
                    ui.gpu_selection_params.rect_min = [pos.x, pos.y];
                    ui.gpu_selection_params.rect_max = [pos.x, pos.y];
                }
                SelectionMode::Brush => {
                    ui.gpu_selection_params.mode = 2; // Brush mode
                    ui.gpu_selection_params.brush_center = [pos.x, pos.y];
                    ui.gpu_selection_params.brush_radius = ui.brush_radius;
                }
                SelectionMode::Slice => {
                    ui.gpu_selection_params.mode = 3; // Slice mode
                    ui.gpu_selection_params.slice_depth = ui.slice_center;
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
                        ui.gpu_selection_params.rect_min = [rect.min.x, rect.min.y];
                        ui.gpu_selection_params.rect_max = [rect.max.x, rect.max.y];
                    }
                }
                SelectionMode::Brush => {
                    ui.gpu_selection_params.brush_center = [pos.x, pos.y];
                    ui.gpu_selection_params.brush_radius = ui.brush_radius;
                }
                SelectionMode::Slice => {
                    ui.gpu_selection_params.slice_depth = ui.slice_center;
                }
            }
        }
    }

    if pointer.primary_released() {
        // Keep selection active - GPU will continue to highlight
        ui.drag_mode  = DragMode::None;
        ui.drag_start = None;
        ui.drag_end   = None;
        ui.is_selecting = false;
        
        // Trigger selection readback for all selection modes
        ui.selection_readback_needed = true;
    }

    // Middle-mouse → move selection on camera plane
    if pointer.button_pressed(egui::PointerButton::Middle) && !ui.selected_indices.is_empty() {
        if let Some(pos) = pointer.press_origin() {
            ui.drag_mode          = DragMode::MovingSelection;
            ui.move_start_mouse   = Some(pos);
            ui.move_start_positions = ui.selected_indices
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
        ui.drag_mode          = DragMode::None;
        ui.move_start_mouse   = None;
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
                if e.button(lbl).clicked() { ui.paused = !ui.paused; }
                if e.button("⏭ Step").clicked() { ui.step_once = true; }
            });
            e.separator();

            e.checkbox(&mut ui.use_gpu_physics, "Use GPU Physics");
            e.separator();

            e.label("Physics");
            let mut changed = false;

            let mut type_count = sim.params.type_count as f32;
            if e.add(Slider::new(&mut type_count, 1.0..=crate::renderer::compute::MAX_TYPES as f32).text("types")).changed() {
                let nc = type_count as usize;
                if nc != sim.params.type_count { sim.set_type_count(nc); }
            }

            let r_max_max = if ui.cap_to_bounds { sim.params.bounds } else { 20.0 };
            changed |= e.add(Slider::new(&mut sim.params.r_max,       0.001..=r_max_max ).text("r_max")).changed();
            changed |= e.add(Slider::new(&mut sim.params.force_scale,  0.0  ..=20.0).text("force scale")).changed();
            changed |= e.add(Slider::new(&mut sim.params.friction,     0.0  ..=1.0 ).text("friction")).changed();
            changed |= e.add(Slider::new(&mut sim.params.dt,           0.0001..=0.1).text("dt")).changed();
            let max_speed_max = if ui.cap_to_bounds { sim.params.bounds } else { 20.0 };
            changed |= e.add(Slider::new(&mut sim.params.max_speed,    0.01 ..=max_speed_max).text("max speed")).changed();
            changed |= e.add(Slider::new(&mut sim.params.beta,         0.01 ..=0.99).text("beta")).changed();
            changed |= e.add(Slider::new(&mut sim.params.particle_size, 0.001..=0.2).text("particle size")).changed();
            changed |= e.add(Slider::new(&mut sim.params.bounds,       0.1  ..=20.0).text("bounds")).changed();
            changed |= e.checkbox(&mut ui.cap_to_bounds, "Cap sliders to bounds").changed();
            changed |= e.checkbox(&mut sim.params.wrap, "Wrap").changed();

            e.separator();
            e.label("CPU step mode");
            e.horizontal(|e| {
                let is_auto  = matches!(sim.params.cpu_step_mode, CpuStepMode::Auto);
                let is_naive = matches!(sim.params.cpu_step_mode, CpuStepMode::Naive);
                let is_grid  = matches!(sim.params.cpu_step_mode, CpuStepMode::GridExact);

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
            if e.add(Slider::new(&mut auto_threshold, 0..=50_000).text("grid threshold")).changed() {
                sim.params.auto_grid_threshold = auto_threshold as usize;
                changed = true;
            }

            let resolved_mode = if sim.last_step_used_grid { "GridExact" } else { "Naive" };
            e.label(format!("Last resolved mode: {}", resolved_mode));
            e.label(format!("Neighbor checks: {}", sim.last_neighbor_checks));
            e.label(format!("Grid resolution: {}", sim.last_grid_res));
            e.label(format!("Last step time: {:.3} ms", sim.last_step_ms));
            e.label(format!("Avg step time: {:.3} ms", sim.avg_step_ms));

            if changed { sim.params_dirty = true; }
            e.separator();

            e.label("Effects");
            
                        
            // Advanced trail modes
            e.label("Advanced Trails:");
            
            // Trail mode dropdown
            egui::ComboBox::from_label("Trail Mode")
                .selected_text(match ui.trace_render_mode {
                    TraceRenderMode::Off => "Off",
                    TraceRenderMode::Simple => "Simple",
                    TraceRenderMode::Lines => "Lines",
                    TraceRenderMode::Dots => "Dots",
                })
                .show_ui(e, |e| {
                    e.selectable_value(&mut ui.trace_render_mode, TraceRenderMode::Off, "Off");
                    e.selectable_value(&mut ui.trace_render_mode, TraceRenderMode::Simple, "Simple");
                    e.selectable_value(&mut ui.trace_render_mode, TraceRenderMode::Lines, "Lines");
                    e.selectable_value(&mut ui.trace_render_mode, TraceRenderMode::Dots, "Dots");
                });
            
            // Trail type filter dropdown
            egui::ComboBox::from_label("Trail Type")
                .selected_text(if ui.trace_type_filter < 0 {
                    "All".to_string()
                } else {
                    format!("Type {}", ui.trace_type_filter)
                })
                .show_ui(e, |e| {
                    e.selectable_value(&mut ui.trace_type_filter, -1, "All");
                    for t in 0..sim.params.type_count {
                        e.selectable_value(&mut ui.trace_type_filter, t as i32, format!("Type {}", t));
                    }
                });
            
            e.add(egui::Slider::new(&mut ui.trace_len, 1..=16).text("Trail Length"));
            #[cfg(debug_assertions)]
            e.checkbox(&mut ui.debug_trails, "Trail Debug Output");
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
                if e.button("+100").clicked() { sim.spawn_random(100); }
                if e.button("+1k").clicked()  { sim.spawn_random(1000); }
                if e.button("+10k").clicked() { sim.spawn_random(10000); }
            });
            e.horizontal(|e| {
                if e.button("Clear").clicked()      { sim.clear_particles(); }
                if e.button("🎲 Rules").clicked()   { sim.randomize_rules(); }
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
                    ui.fly_pos   = Vec3::new(
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
                e.add(Slider::new(&mut ui.fly_yaw,   -3.14..=3.14).text("yaw"));
                e.add(Slider::new(&mut ui.fly_pitch,  -1.5 ..=1.5 ).text("pitch"));
            }
            e.separator();

            e.label("Selection");
            e.horizontal(|e| {
                if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Rect),  "Rect" ).clicked() { ui.selection_mode = SelectionMode::Rect;  }
                if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Brush), "Brush").clicked() { ui.selection_mode = SelectionMode::Brush; }
                if e.selectable_label(matches!(ui.selection_mode, SelectionMode::Slice), "Slice").clicked() { ui.selection_mode = SelectionMode::Slice; }
            });
            if matches!(ui.selection_mode, SelectionMode::Slice) {
                e.add(Slider::new(&mut ui.slice_center,    -20.0..=20.0).text("slice center"));
                e.add(Slider::new(&mut ui.slice_thickness,  0.001..=5.0 ).text("slice thickness"));
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
                    if e.button("Cool down").clicked()    { sim.scale_velocities(0.1); }
                    if e.button("Energy boost").clicked() { sim.scale_velocities(2.0); }
                    if e.button("Freeze").clicked()       { sim.scale_velocities(0.0); }
                });
                e.horizontal(|e| {
                    e.label("Assign type:");
                    let mut tv = ui.pending_assign_type as usize;
                    if e.add(Slider::new(&mut tv, 0..=sim.params.type_count.saturating_sub(1)).text("")).changed() {
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

// ── Rule matrix panel ─────────────────────────────────────────────────────────
fn draw_rule_matrix(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    let n  = sim.params.type_count;
    let dt = ctx.input(|i| i.stable_dt);

    // Advance hold-click ramp (150 ms delay, then 1.0 unit/s)
    if let Some((row, col, sign)) = ui.matrix_hold {
        ui.matrix_hold_timer += dt;
        if ui.matrix_hold_timer > 0.15 {
            let val = sim.get_rule(row, col);
            sim.set_rule(row, col, (val + sign * dt).clamp(-1.0, 1.0));
        }
    }

    egui::Window::new("Rule Matrix")
        .default_pos([350.0, 12.0])
        .resizable(false)
        .show(ctx, |e| {
            e.label("LMB = +  RMB = −  hold to ramp  scroll = fine");
            e.separator();

            e.horizontal(|e| {
                if e.button("Zero all").clicked() {
                    for i in 0..n { for j in 0..n { sim.set_rule(i, j, 0.0); } }
                }
                if e.button("Symmetrize").clicked() {
                    for i in 0..n {
                        for j in (i+1)..n {
                            let avg = (sim.get_rule(i, j) + sim.get_rule(j, i)) * 0.5;
                            sim.set_rule(i, j, avg);
                            sim.set_rule(j, i, avg);
                        }
                    }
                }
                if e.button("Invert").clicked() {
                    for i in 0..n { for j in 0..n {
                        let v = sim.get_rule(i, j);
                        sim.set_rule(i, j, -v);
                    }}
                }
            });
            e.separator();

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
                        if resp.clicked()           { sim.set_rule(row, col, (val + 0.1).clamp(-1.0, 1.0)); }
                        if resp.secondary_clicked() { sim.set_rule(row, col, (val - 0.1).clamp(-1.0, 1.0)); }

                        // Hold tracking
                        if resp.is_pointer_button_down_on() {
                            let sign = if ctx.input(|i| i.pointer.secondary_down()) { -1.0_f32 } else { 1.0_f32 };
                            if ui.matrix_hold != Some((row, col, sign)) {
                                ui.matrix_hold       = Some((row, col, sign));
                                ui.matrix_hold_timer = 0.0;
                            }
                        } else if matches!(ui.matrix_hold, Some((r, c, _)) if r == row && c == col) {
                            ui.matrix_hold       = None;
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

            e.separator();
            e.horizontal(|e| {
                e.colored_label(Color32::from_rgb(30, 220, 30), "■ attract");
                e.colored_label(Color32::from_rgb(220, 30, 30), "■ repel");
            });
        });
}

// ── Reaction Matrix panel ───────────────────────────────────────────────────────
fn draw_reaction_matrix(ctx: &Context, sim: &mut SimState, ui: &mut UiState) {
    let n  = sim.params.type_count;
    let _dt = ctx.input(|i| i.stable_dt);

    egui::Window::new("Reaction Matrix")
        .default_pos([650.0, 12.0])
        .resizable(false)
        .show(ctx, |e| {
            e.label("LMB = cycle result  RMB = no reaction");
            e.separator();

            // Reaction controls
            e.horizontal(|e| {
                e.checkbox(&mut sim.params.reactions_enabled, "Enable Reactions");
                if e.button("Default").clicked() {
                    sim.default_reaction_table();
                }
                if e.button("Clear").clicked() {
                    sim.resize_reaction_table();
                }
            });
            
            e.horizontal(|e| {
                let mix_radius_max = if ui.cap_to_bounds { sim.params.bounds } else { 20.0 };
                e.add(Slider::new(&mut sim.params.mix_radius, 0.01..=mix_radius_max).text("mix radius"));
                e.add(Slider::new(&mut sim.params.reaction_probability, 0.01..=1.0).text("probability"));
            });
            
            e.checkbox(&mut sim.params.preserve_particle_count, "Preserve particle count");
            e.separator();

            Grid::new("reaction_matrix").spacing([2.0, 2.0]).show(e, |e| {
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

                        // LMB: cycle through result types
                        if resp.clicked() {
                            let current = sim.rx(row, col);
                            let next = if current < 0 { 0 } else { (current + 1) % (n as i32) };
                            sim.reaction_table[row * n + col] = next;
                        }

                        // RMB: set to -1 (no reaction)
                        if resp.secondary_clicked() {
                            sim.reaction_table[row * n + col] = -1;
                        }

                        // Scroll fine-tune
                        if resp.hovered() {
                            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
                            if scroll != 0.0 {
                                let current = sim.rx(row, col);
                                let new_val = (current + scroll.signum() as i32).clamp(-1, (n - 1) as i32);
                                sim.reaction_table[row * n + col] = new_val;
                            }
                        }
                    }
                    e.end_row();
                }
            });

            e.separator();
            e.horizontal(|e| {
                e.colored_label(Color32::from_rgb(60, 60, 60), "■ no reaction");
                e.label("= particle transforms to shown type");
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
                    if e.button("🗑").clicked() { to_remove = Some(i); }
                });
            }

            if let Some(i) = to_remove {
                sim.book.remove_prefab(i);
                if ui.selected_prefab == Some(i) { ui.selected_prefab = None; }
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

    let selected: Vec<_> = ui.selected_indices.iter()
        .filter_map(|&idx| sim.particles.get(idx))
        .collect();
    
    if selected.is_empty() {
        log::warn!("Selected particles not found in simulation");
        return;
    }

    let com: Vec3 = selected.iter()
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
        Color32::from_rgb(245,  92,  92),
        Color32::from_rgb( 92, 194, 245),
        Color32::from_rgb(143, 245,  92),
        Color32::from_rgb(245, 194,  66),
        Color32::from_rgb(220, 117, 245),
        Color32::from_rgb(245, 143,  66),
        Color32::from_rgb( 92, 245, 194),
        Color32::from_rgb(245, 194, 220),
    ];
    colors[kind % colors.len()]
}