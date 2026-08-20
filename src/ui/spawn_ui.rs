//! Spawn panel — particle population as an editable composition.
//!
//! Replaces the ±100/±1k/+10k buttons. You set a pool of *available particles*,
//! then split it across types in a donut chart: click a wedge and type an exact
//! count, or drag it. Whatever's unallocated shows as a grey wedge rather than
//! being silently normalized away, so the numbers always add up in front of you.
//!
//! Every committed edit does a full respawn, which is what makes the chart
//! honest — the population on screen is always exactly what the chart says.
//! Drags respawn on release rather than per-frame, otherwise a single gesture
//! would rebuild 50k particles sixty times a second.

use crate::sim::{SimState, MAX_CPU_PHYSICS_PARTICLES, MAX_RENDER_PARTICLES};
use crate::ui::UiState;
use egui::{Color32, RichText};

use super::type_color_egui;

const TAU: f32 = std::f32::consts::TAU;

pub fn particle_cap() -> usize {
    MAX_RENDER_PARTICLES.min(MAX_CPU_PHYSICS_PARTICLES)
}

/// Keep the mix vectors in step with `type_count`. Cheap; call every frame.
///
/// Each vector is checked independently on purpose. Gating `mix_locked` behind
/// `type_mix`'s length is a trap: `adopt_current` rewrites `type_mix` to the
/// right length on its own, which makes the outer check pass while `mix_locked`
/// is still empty — and then the legend indexes into a zero-length vec.
pub fn sync_mix(sim: &SimState, ui: &mut UiState) {
    let n = sim.params.type_count;
    if ui.type_mix.len() != n {
        ui.type_mix.resize(n, 0);
    }
    if ui.mix_locked.len() != n {
        ui.mix_locked.resize(n, false);
    }
    if ui.mix_editing.map_or(false, |i| i >= n) {
        ui.mix_editing = None;
    }
    if ui.mix_drag.map_or(false, |i| i >= n) {
        ui.mix_drag = None;
    }
}

/// Read the live population back into the chart.
pub fn adopt_current(sim: &SimState, ui: &mut UiState) {
    let n = sim.params.type_count;
    ui.type_mix = vec![0usize; n];
    ui.mix_locked.resize(n, false);
    for p in &sim.particles {
        let k = p.kind as usize;
        if k < n {
            ui.type_mix[k] += 1;
        }
    }
    ui.particle_pool = ui.particle_pool.max(allocated(ui));
}

pub fn allocated(ui: &UiState) -> usize {
    ui.type_mix.iter().sum()
}

fn commit(sim: &mut SimState, ui: &mut UiState) {
    let mix = ui.type_mix.clone();
    sim.respawn_mix(&mix);
    ui.selected_indices.clear();
    ui.clear_selection_requested = true;
}

// ── Panel ───────────────────────────────────────────────────────────────────

pub fn draw_spawn_panel(e: &mut egui::Ui, sim: &mut SimState, ui: &mut UiState) {
    sync_mix(sim, ui);
    let n = sim.params.type_count;
    if n == 0 {
        return;
    }
    let cap = particle_cap();

    e.horizontal(|e| {
        e.label("Available");
        let mut pool = ui.particle_pool;
        if e
            .add(
                egui::DragValue::new(&mut pool)
                    .speed(50.0)
                    .clamp_range(0..=cap),
            )
            .on_hover_text(format!("Total pool to draw from (cap {cap})"))
            .changed()
        {
            ui.particle_pool = pool;
            // Shrinking the pool trims the largest unlocked types first.
            while allocated(ui) > ui.particle_pool {
                let over = allocated(ui) - ui.particle_pool;
                let Some(idx) = largest_unlocked(ui, None) else { break };
                let take = over.min(ui.type_mix[idx]);
                if take == 0 {
                    break;
                }
                ui.type_mix[idx] -= take;
            }
        }

        let used = allocated(ui);
        let free = ui.particle_pool.saturating_sub(used);
        e.label(
            RichText::new(format!("{used} placed · {free} free"))
                .weak()
                .small(),
        );
    });

    e.horizontal(|e| {
        if e.button("Even split")
            .on_hover_text("Divide the whole pool across unlocked types")
            .clicked()
        {
            even_split(ui, n);
            commit(sim, ui);
            ui.flash("Respawned: even split");
        }
        if e.button("Fill free")
            .on_hover_text("Hand the unallocated remainder to unlocked types")
            .clicked()
        {
            fill_free(ui, n);
            commit(sim, ui);
            ui.flash("Respawned");
        }
        if e.button("Match current")
            .on_hover_text("Read the live population back into the chart")
            .clicked()
        {
            adopt_current(sim, ui);
            ui.flash("Chart matched to population");
        }
        if e.button("Clear").clicked() {
            for v in &mut ui.type_mix {
                *v = 0;
            }
            commit(sim, ui);
            ui.flash("Cleared");
        }
        if e.button("↻ Respawn")
            .on_hover_text("Rebuild the population from the chart")
            .clicked()
        {
            commit(sim, ui);
            ui.flash("Respawned");
        }
    });

    draw_donut(e, sim, ui, n);
    draw_legend(e, sim, ui, n);
}

// ── Donut ───────────────────────────────────────────────────────────────────

fn draw_donut(e: &mut egui::Ui, sim: &mut SimState, ui: &mut UiState, n: usize) {
    let size = 168.0;
    let (rect, resp) = e.allocate_exact_size(egui::vec2(size, size), egui::Sense::click_and_drag());
    let center = rect.center();
    let r_out = size * 0.46;
    let r_in = size * 0.26;

    if ui.type_mix.len() < n || ui.mix_locked.len() < n {
        return; // sync_mix hasn't caught up with a type_count change yet
    }
    let pool = ui.particle_pool.max(1);
    let used = allocated(ui);
    let free = pool.saturating_sub(used);
    let total = (used + free).max(1) as f32;

    // Which wedge is the pointer in?
    let pointer = e.input(|i| i.pointer.interact_pos());
    let hover_idx = pointer.and_then(|p| wedge_at(p, center, r_in, r_out, &ui.type_mix, free, n));

    if e.is_rect_visible(rect) {
        let painter = e.painter();
        painter.circle_filled(center, r_out, Color32::from_gray(30));

        let mut a = 0.0f32; // polar() already anchors 0 at 12 o'clock
        for t in 0..n {
            let frac = ui.type_mix[t] as f32 / total;
            if frac <= 0.0 {
                continue;
            }
            let sweep = frac * TAU;
            let hot = hover_idx == Some(t) || ui.mix_editing == Some(t);
            let mut col = type_color_egui(t);
            if !hot {
                col = col.gamma_multiply(0.82);
            }
            fill_ring(painter, center, r_in, r_out, a, a + sweep, col);
            if ui.mix_editing == Some(t) {
                stroke_ring(painter, center, r_in, r_out, a, a + sweep, Color32::WHITE);
            }
            a += sweep;
        }

        // Unallocated remainder — grey, honest, and clickable to hand it out.
        if free > 0 {
            let sweep = free as f32 / total * TAU;
            let hot = hover_idx == Some(n);
            let col = if hot {
                Color32::from_gray(88)
            } else {
                Color32::from_gray(62)
            };
            fill_ring(painter, center, r_in, r_out, a, a + sweep, col);
        }

        painter.circle_filled(center, r_in, Color32::from_gray(24));

        // Center readout
        painter.text(
            center - egui::vec2(0.0, 7.0),
            egui::Align2::CENTER_CENTER,
            format!("{used}"),
            egui::FontId::monospace(16.0),
            Color32::from_gray(235),
        );
        painter.text(
            center + egui::vec2(0.0, 9.0),
            egui::Align2::CENTER_CENTER,
            "particles",
            egui::FontId::proportional(9.0),
            Color32::from_gray(150),
        );
    }

    // ── Interaction ─────────────────────────────────────────────────────────
    if resp.drag_started() {
        ui.mix_drag = hover_idx.filter(|i| *i < n && !locked(ui, *i));
    }

    if let Some(idx) = ui.mix_drag {
        if resp.dragged() {
            // Vertical drag, same gesture language as the matrix cells.
            let step = (ui.particle_pool as f32 / 400.0).max(1.0);
            let delta = -resp.drag_delta().y * step;
            adjust(ui, idx, delta.round() as i64, n);
            e.output_mut(|o| o.cursor_icon = egui::CursorIcon::ResizeVertical);
        }
    }

    if resp.drag_released() {
        if ui.mix_drag.is_some() {
            commit(sim, ui);
            ui.mix_drag = None;
            ui.flash("Respawned");
        }
    }

    if resp.clicked() && ui.mix_drag.is_none() {
        match hover_idx {
            Some(i) if i < n => {
                ui.mix_editing = Some(i);
                ui.mix_edit_buf = ui.type_mix[i].to_string();
            }
            Some(_) => {
                // Clicked the free wedge — hand it out.
                fill_free(ui, n);
                commit(sim, ui);
                ui.flash("Respawned");
            }
            None => ui.mix_editing = None,
        }
    }

    if let Some(i) = hover_idx {
        let txt = if i < n {
            format!(
                "Type {i}: {} ({:.1}%)\nclick to type a count · drag ↕ to adjust",
                ui.type_mix[i],
                ui.type_mix[i] as f32 / total * 100.0
            )
        } else {
            format!("Unallocated: {free}\nclick to hand out to unlocked types")
        };
        resp.on_hover_text(txt);
    }

    // ── Inline count entry ──────────────────────────────────────────────────
    if let Some(i) = ui.mix_editing {
        if i < n {
            e.horizontal(|e| {
                e.colored_label(type_color_egui(i), RichText::new(format!("T{i}")).strong());
                let te = e.add(
                    egui::TextEdit::singleline(&mut ui.mix_edit_buf)
                        .desired_width(80.0)
                        .hint_text("count"),
                );
                te.request_focus();

                let submit = te.lost_focus() && e.input(|k| k.key_pressed(egui::Key::Enter));
                let apply = e.button("Set").clicked() || submit;

                if apply {
                    if let Ok(want) = ui.mix_edit_buf.trim().parse::<usize>() {
                        set_count(ui, i, want, n);
                        commit(sim, ui);
                        ui.flash(format!("Respawned — T{i} = {}", ui.type_mix[i]));
                    }
                    ui.mix_editing = None;
                }
                if e.button("Cancel").clicked() {
                    ui.mix_editing = None;
                }
            });
        }
    }
}

fn draw_legend(e: &mut egui::Ui, sim: &mut SimState, ui: &mut UiState, n: usize) {
    let mut dirty = false;
    egui::Grid::new("mix_legend")
        .num_columns(4)
        .spacing([8.0, 3.0])
        .show(e, |e| {
            for t in 0..n {
                let (dot, _) = e.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                e.painter()
                    .circle_filled(dot.center(), 5.0, type_color_egui(t));

                e.label(format!("T{t}"));

                let mut v = ui.type_mix.get(t).copied().unwrap_or(0);
                let is_locked = locked(ui, t);
                let head = ui.particle_pool.saturating_sub(allocated(ui)) + v;
                let resp = e.add_enabled(
                    !is_locked,
                    egui::DragValue::new(&mut v)
                        .speed(10.0)
                        .clamp_range(0..=head),
                );
                if resp.changed() {
                    if let Some(slot) = ui.type_mix.get_mut(t) {
                        *slot = v;
                    }
                }
                // Respawn on release, not on every frame of a drag — a full
                // rebuild of 50k particles at 60 fps would be unusable.
                if resp.drag_released() || resp.lost_focus() {
                    dirty = true;
                }

                let lock_label = if is_locked { "🔒" } else { "🔓" };
                if e.button(lock_label)
                    .on_hover_text("Locked types are never auto-adjusted")
                    .clicked()
                {
                    if let Some(slot) = ui.mix_locked.get_mut(t) {
                        *slot = !is_locked;
                    }
                }
                e.end_row();
            }
        });

    if dirty {
        commit(sim, ui);
    }
}

// ── Mix math ────────────────────────────────────────────────────────────────

fn locked(ui: &UiState, i: usize) -> bool {
    ui.mix_locked.get(i).copied().unwrap_or(false)
}

fn largest_unlocked(ui: &UiState, skip: Option<usize>) -> Option<usize> {
    ui.type_mix
        .iter()
        .enumerate()
        .filter(|(i, _)| !locked(ui, *i) && Some(*i) != skip)
        .max_by_key(|(_, v)| **v)
        .filter(|(_, v)| **v > 0)
        .map(|(i, _)| i)
}

/// Set one type to an exact count, taking from free space first and stealing
/// from the largest unlocked neighbour only if the pool is already full.
fn set_count(ui: &mut UiState, idx: usize, want: usize, _n: usize) {
    if idx >= ui.type_mix.len() {
        return;
    }
    let want = want.min(ui.particle_pool);
    let cur = ui.type_mix[idx];
    if want > cur {
        adjust(ui, idx, (want - cur) as i64, _n);
    } else {
        ui.type_mix[idx] = want;
    }
}

fn adjust(ui: &mut UiState, idx: usize, delta: i64, n: usize) {
    if idx >= n || idx >= ui.type_mix.len() || delta == 0 {
        return;
    }
    if delta < 0 {
        let d = (-delta) as usize;
        ui.type_mix[idx] = ui.type_mix[idx].saturating_sub(d);
        return;
    }

    let mut want = delta as usize;
    let free = ui.particle_pool.saturating_sub(allocated(ui));
    let from_free = want.min(free);
    ui.type_mix[idx] += from_free;
    want -= from_free;

    // Pool exhausted — steal from the largest unlocked other type.
    while want > 0 {
        let Some(donor) = largest_unlocked(ui, Some(idx)) else { break };
        let take = want.min(ui.type_mix[donor]);
        if take == 0 {
            break;
        }
        ui.type_mix[donor] -= take;
        ui.type_mix[idx] += take;
        want -= take;
    }
}

fn even_split(ui: &mut UiState, n: usize) {
    let n = n.min(ui.type_mix.len());
    let locked_total: usize = (0..n)
        .filter(|i| locked(ui, *i))
        .filter_map(|i| ui.type_mix.get(i).copied())
        .sum();
    let open: Vec<usize> = (0..n).filter(|i| !locked(ui, *i)).collect();
    if open.is_empty() {
        return;
    }
    let budget = ui.particle_pool.saturating_sub(locked_total);
    let each = budget / open.len();
    let mut rem = budget % open.len();
    for i in open {
        ui.type_mix[i] = each + if rem > 0 { rem -= 1; 1 } else { 0 };
    }
}

fn fill_free(ui: &mut UiState, n: usize) {
    let n = n.min(ui.type_mix.len());
    let free = ui.particle_pool.saturating_sub(allocated(ui));
    if free == 0 {
        return;
    }
    let open: Vec<usize> = (0..n).filter(|i| !locked(ui, *i)).collect();
    if open.is_empty() {
        return;
    }
    let each = free / open.len();
    let mut rem = free % open.len();
    for i in open {
        ui.type_mix[i] += each + if rem > 0 { rem -= 1; 1 } else { 0 };
    }
}

// ── Geometry ────────────────────────────────────────────────────────────────

/// Which wedge contains `p`? Returns `n` for the unallocated remainder.
fn wedge_at(
    p: egui::Pos2,
    center: egui::Pos2,
    r_in: f32,
    r_out: f32,
    mix: &[usize],
    free: usize,
    n: usize,
) -> Option<usize> {
    let d = p - center;
    let r = d.length();
    if r < r_in || r > r_out {
        return None;
    }
    // Normalize to 0..TAU measured from 12 o'clock, clockwise.
    let mut ang = d.y.atan2(d.x) + std::f32::consts::FRAC_PI_2;
    while ang < 0.0 {
        ang += TAU;
    }
    while ang >= TAU {
        ang -= TAU;
    }

    let total = (mix.iter().sum::<usize>() + free).max(1) as f32;
    let mut acc = 0.0;
    for t in 0..n {
        let sweep = mix[t] as f32 / total * TAU;
        if sweep > 0.0 && ang >= acc && ang < acc + sweep {
            return Some(t);
        }
        acc += sweep;
    }
    if free > 0 && ang >= acc {
        return Some(n);
    }
    None
}

/// Fill an annulus segment as one mesh.
///
/// Drawing it as a strip of separate quads leaves a visible spoke at every
/// shared edge — each quad anti-aliases against the background independently,
/// so the seams show as radial lines. A single mesh with shared vertices has no
/// internal edges to feather.
fn fill_ring(
    painter: &egui::Painter,
    center: egui::Pos2,
    r_in: f32,
    r_out: f32,
    a0: f32,
    a1: f32,
    color: Color32,
) {
    let span = a1 - a0;
    if span <= 0.0 {
        return;
    }
    let steps = ((span / 0.10).ceil() as usize).clamp(1, 256);
    let da = span / steps as f32;

    let mut mesh = egui::Mesh::default();
    for s in 0..=steps {
        let b = a0 + da * s as f32;
        mesh.colored_vertex(polar(center, r_in, b), color);
        mesh.colored_vertex(polar(center, r_out, b), color);
    }
    for s in 0..steps {
        let i = (s as u32) * 2;
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i + 1, i + 3, i + 2);
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn stroke_ring(
    painter: &egui::Painter,
    center: egui::Pos2,
    r_in: f32,
    r_out: f32,
    a0: f32,
    a1: f32,
    color: Color32,
) {
    let stroke = egui::Stroke::new(1.5, color);
    let steps = (((a1 - a0) / 0.10).ceil() as usize).clamp(1, 256);
    let da = (a1 - a0) / steps as f32;
    let mut outer = Vec::with_capacity(steps + 1);
    let mut inner = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let b = a0 + da * s as f32;
        outer.push(polar(center, r_out, b));
        inner.push(polar(center, r_in, b));
    }
    painter.add(egui::Shape::line(outer, stroke));
    painter.add(egui::Shape::line(inner, stroke));
    painter.add(egui::Shape::line(
        vec![polar(center, r_in, a0), polar(center, r_out, a0)],
        stroke,
    ));
    painter.add(egui::Shape::line(
        vec![polar(center, r_in, a1), polar(center, r_out, a1)],
        stroke,
    ));
}

/// Angles run clockwise from 12 o'clock.
fn polar(center: egui::Pos2, r: f32, ang: f32) -> egui::Pos2 {
    let a = ang - std::f32::consts::FRAC_PI_2;
    egui::pos2(center.x + r * a.cos(), center.y + r * a.sin())
}
