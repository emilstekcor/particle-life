//! Audio tab of the Matrix Editor.
//!
//! Every channel you add is a band with its own N×N matrix, edited with the
//! same gestures as the force matrix: drag ↕ to set, scroll to fine-tune,
//! double-click to zero, right-click to flip sign. A cell is the weight for
//! that type pair; zero means the band doesn't touch it.
//!
//! Combine mode (Add / Scale) and swing mode (Bipolar / Ping-pong / Springy)
//! are latching push buttons — picking one pops the other out.

use crate::audio::{
    AudioMod, BandLayer, CombineMode, GateState, LayerTarget, SwingMode, BANDS, BAND_NAMES,
};
use crate::sim::SimState;
use crate::ui::UiState;
use egui::{Color32, Context, Grid, RichText};

// Private helpers from the parent `ui` module — a child module can see them.
use super::{effective_cell_size, matrix_cell, take_scroll, type_header};

/// Tab entry point. Signature matches the other `draw_*_matrix_content` fns.
pub fn draw_audio_matrix_content(
    ctx: &Context,
    sim: &mut SimState,
    ui: &mut UiState,
    e: &mut egui::Ui,
) {
    let n = sim.params.type_count;
    let cell = effective_cell_size(ui.matrix_cell_size, n);
    let mut status: Option<String> = None;

    // Borrow audio separately from the rest of UiState so the closures below
    // don't fight over `ui`.
    {
        let audio = &mut ui.audio;

        draw_source(e, sim, audio, &mut status);
        if !audio.has_file() {
            if let Some(msg) = status {
                ui.flash(msg);
            }
            return;
        }

        e.separator();
        draw_transport(e, audio);
        e.separator();
        draw_arm_and_shape(e, sim, audio, &mut status);
        e.separator();
        draw_band_strip(e, audio, n, &mut status);
        e.separator();
        draw_channels(ctx, e, sim, audio, n, cell, &mut status);
    }

    if let Some(msg) = status {
        ui.flash(msg);
    }
}

// ── Source ──────────────────────────────────────────────────────────────────

fn draw_source(
    e: &mut egui::Ui,
    sim: &mut SimState,
    audio: &mut AudioMod,
    status: &mut Option<String>,
) {
    e.horizontal(|e| {
        if e.button("Load audio…").clicked() {
            if audio.pick_file() {
                *status = Some(format!("Loaded {}", audio.file_name));
            } else if let Some(err) = &audio.load_error {
                *status = Some(err.clone());
            }
        }
        if audio.has_file() {
            e.label(RichText::new(&audio.file_name).strong());
            if e.button("✖").on_hover_text("Unload").clicked() {
                audio.disarm(sim);
                audio.stop();
                // Keep the authored layers; only the source goes away.
                let layers = std::mem::take(&mut audio.layers);
                *audio = AudioMod::new();
                audio.layers = layers;
                *status = Some("Audio unloaded".into());
            }
        } else {
            e.label(RichText::new("no file").weak());
        }
    });

    if let Some(err) = &audio.load_error {
        e.colored_label(Color32::from_rgb(220, 120, 90), err);
    }

    if !audio.has_file() {
        e.label(
            RichText::new("mp3 · wav · flac · ogg · m4a · mp4 (audio track)")
                .weak()
                .small(),
        );
    }
}

// ── Transport ───────────────────────────────────────────────────────────────

fn draw_transport(e: &mut egui::Ui, audio: &mut AudioMod) {
    e.horizontal(|e| {
        let label = if audio.playing { "⏸ Pause" } else { "▶ Play" };
        if e.button(label).clicked() {
            audio.toggle_play();
        }
        if e.button("⏹ Stop").clicked() {
            audio.stop();
        }
        e.checkbox(&mut audio.looping, "loop");
        e.checkbox(&mut audio.muted, "mute")
            .on_hover_text("Silences output; analysis keeps running");
        e.add(
            egui::Slider::new(&mut audio.volume, 0.0..=1.0)
                .show_value(false)
                .text("vol"),
        );

        let dur = audio.duration_secs();
        let mut pos = audio.position_secs();
        e.label(format!("{} / {}", fmt_time(pos), fmt_time(dur)));
        if e
            .add(
                egui::Slider::new(&mut pos, 0.0..=dur.max(0.01))
                    .show_value(false)
                    .text(""),
            )
            .drag_released()
        {
            audio.seek_secs(pos);
        }
    });
}

// ── Arm, global shape, base matrix ──────────────────────────────────────────

fn draw_arm_and_shape(
    e: &mut egui::Ui,
    sim: &mut SimState,
    audio: &mut AudioMod,
    status: &mut Option<String>,
) {
    e.horizontal(|e| {
        let mut armed = audio.armed;
        if e.checkbox(&mut armed, "Drive force matrix")
            .on_hover_text(
                "Snapshots the current matrix and modulates around it.\n\
                 Unchecking restores the snapshot exactly.",
            )
            .changed()
        {
            if armed {
                audio.arm(sim);
                *status = Some("Matrix snapshot captured".into());
            } else {
                audio.disarm(sim);
                *status = Some("Matrix restored".into());
            }
        }

        if audio.armed
            && e.button("Re-snapshot")
                .on_hover_text("Take a new base from the current matrix")
                .clicked()
        {
            audio.capture_base(sim);
            *status = Some("New base captured".into());
        }

        e.checkbox(&mut audio.show_base, "show base")
            .on_hover_text("Display the snapshot you're modulating around");
    });

    e.add(
        egui::Slider::new(&mut audio.depth, 0.0..=1.0)
            .text("depth")
            .fixed_decimals(2),
    )
    .on_hover_text("Global scale on every layer's drive. Start small.");

    e.add(
        egui::Slider::new(&mut audio.smoothing, 0.0..=0.98)
            .text("smoothing")
            .fixed_decimals(2),
    )
    .on_hover_text("Slew limiting on the final write. 0 is raw, ~0.85 glides.");

    e.horizontal(|e| {
        e.add(
            egui::Slider::new(&mut audio.attack, 0.0005..=0.2)
                .logarithmic(true)
                .text("attack s"),
        );
        e.add(
            egui::Slider::new(&mut audio.release, 0.02..=1.5)
                .logarithmic(true)
                .text("release s"),
        );
    });

    if audio.show_base && audio.armed {
        draw_base_readout(e, audio, sim.params.type_count);
    }
}

/// Read-only view of the snapshot, so you can see what you're swinging around.
fn draw_base_readout(e: &mut egui::Ui, audio: &AudioMod, n: usize) {
    let base = audio.base_snapshot();
    if base.len() != n * n {
        return;
    }
    e.label(RichText::new("base snapshot").weak().small());
    Grid::new("audio_base_readout")
        .spacing([2.0, 2.0])
        .show(e, |e| {
            e.label("");
            for c in 0..n {
                type_header(e, c, 18.0, false);
            }
            e.end_row();
            for r in 0..n {
                type_header(e, r, 18.0, false);
                for c in 0..n {
                    let v = base[r * n + c];
                    matrix_cell(e, 18.0, value_color(v), None, egui::Sense::hover(), false)
                        .on_hover_text(format!("T{r} → T{c} base: {v:+.3}"));
                }
                e.end_row();
            }
        });
}

// ── Band meters ─────────────────────────────────────────────────────────────

/// Meters double as the channel picker — click one to add that band.
fn draw_band_strip(
    e: &mut egui::Ui,
    audio: &mut AudioMod,
    n: usize,
    status: &mut Option<String>,
) {
    e.horizontal(|e| {
        e.label(RichText::new("bands").weak());
        e.label(
            RichText::new("click a meter to add it as a channel")
                .weak()
                .small(),
        );
    });

    let mut add: Option<usize> = None;
    e.horizontal(|e| {
        for b in 0..BANDS {
            let used = audio.layers.iter().any(|l| l.band == b);
            if draw_band_meter(e, audio.bands[b], BAND_NAMES[b], b, used).clicked() && !used {
                add = Some(b);
            }
        }
    });

    if let Some(b) = add {
        audio.layers.push(BandLayer::new(b, n));
        *status = Some(format!("Added {} channel", BAND_NAMES[b]));
    }
}

fn draw_band_meter(
    e: &mut egui::Ui,
    v: f32,
    name: &str,
    idx: usize,
    used: bool,
) -> egui::Response {
    let w = 26.0;
    let h = 54.0;
    let (rect, resp) = e.allocate_exact_size(egui::vec2(w, h), egui::Sense::click());
    if e.is_rect_visible(rect) {
        let p = e.painter();
        p.rect_filled(rect, 3.0, Color32::from_gray(38));
        let fill_h = h * v.clamp(0.0, 1.0);
        let fill_rect =
            egui::Rect::from_min_max(egui::pos2(rect.min.x, rect.max.y - fill_h), rect.max);
        p.rect_filled(fill_rect, 3.0, band_color(idx));
        if used {
            p.rect_stroke(rect, 3.0, egui::Stroke::new(1.5, Color32::from_gray(210)));
        } else if resp.hovered() {
            p.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, Color32::from_gray(140)));
        }
    }
    let tip = if used {
        format!("{name}: {v:.2}\nalready a channel")
    } else {
        format!("{name}: {v:.2}\nclick to add as a channel")
    };
    resp.on_hover_text(tip)
}

/// Hue walks the spectrum so layers are distinguishable at a glance.
pub fn band_color(idx: usize) -> Color32 {
    let t = idx as f32 / (BANDS - 1).max(1) as f32;
    Color32::from_rgb(
        (60.0 + 190.0 * t) as u8,
        (200.0 - 90.0 * t) as u8,
        (220.0 - 140.0 * t) as u8,
    )
}

// ── Channels ────────────────────────────────────────────────────────────────

fn draw_channels(
    ctx: &Context,
    e: &mut egui::Ui,
    sim: &SimState,
    audio: &mut AudioMod,
    n: usize,
    cell: f32,
    status: &mut Option<String>,
) {
    e.horizontal(|e| {
        e.label(RichText::new("channels").weak());

        egui::ComboBox::from_id_source("add_channel")
            .selected_text("+ add channel")
            .width(140.0)
            .show_ui(e, |e| {
                for b in 0..BANDS {
                    let used = audio.layers.iter().any(|l| l.band == b);
                    let label = if used {
                        format!("{} (added)", BAND_NAMES[b])
                    } else {
                        BAND_NAMES[b].to_string()
                    };
                    if e.add_enabled(!used, egui::Button::new(label)).clicked() {
                        audio.layers.push(BandLayer::new(b, n));
                    }
                }
            });

        if e
            .button("auto-fill")
            .on_hover_text(
                "One channel per band, each seeded with an alternating-sign\n\
                 off-diagonal pattern. A starting point, not a preset.",
            )
            .clicked()
        {
            audio.auto_fill(n);
            *status = Some(format!("{} channels seeded", audio.layers.len()));
        }

        if e.button("clear all").clicked() {
            audio.layers.clear();
            *status = Some("Channels cleared".into());
        }
    });

    if audio.layers.is_empty() {
        e.label(
            RichText::new("No channels yet — click a band meter above.")
                .weak()
                .small(),
        );
        return;
    }

    let mut remove: Option<usize> = None;
    let mut hovered_now: Option<(usize, usize, usize)> = None;
    let hovered_last = audio.hovered;

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .drag_to_scroll(false) // cell drags adjust values, not scroll position
        .max_height(520.0)
        .id_source("audio_channels")
        .show(e, |e| {
            for li in 0..audio.layers.len() {
                let band = audio.layers[li].band;
                let level = audio.bands[band];
                let tint = band_color(band);

                egui::Frame::group(e.style())
                    .stroke(egui::Stroke::new(1.0, tint.gamma_multiply(0.55)))
                    .show(e, |e| {
                        // ── Header ──────────────────────────────────────────
                        e.horizontal(|e| {
                            let layer = &mut audio.layers[li];
                            e.checkbox(&mut layer.enabled, "");

                            let arrow = if layer.open { "▼" } else { "▶" };
                            if e.button(arrow).clicked() {
                                layer.open = !layer.open;
                            }

                            e.colored_label(tint, RichText::new(BAND_NAMES[band]).strong());
                            level_bar(e, level, tint);

                            e.add(
                                egui::DragValue::new(&mut layer.gain)
                                    .speed(0.02)
                                    .clamp_range(-2.0..=2.0)
                                    .fixed_decimals(2)
                                    .prefix("×"),
                            )
                            .on_hover_text("Layer gain — scales every cell");

                            e.label(
                                RichText::new(format!("{} cells", layer.active_cells()))
                                    .weak()
                                    .small(),
                            );

                            e.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |e| {
                                    if e.button("🗑").on_hover_text("Remove channel").clicked() {
                                        remove = Some(li);
                                    }
                                    if e.button("zero").on_hover_text("Clear this matrix").clicked()
                                    {
                                        layer.clear();
                                    }
                                },
                            );
                        });

                        if !audio.layers[li].open {
                            return;
                        }

                        // ── Mode buttons (exclusive latching) ───────────────
                        e.horizontal(|e| {
                            let layer = &mut audio.layers[li];

                            e.label(RichText::new("drives").weak().small());
                            for t in LayerTarget::ALL {
                                if e
                                    .selectable_label(layer.target == t, t.label())
                                    .on_hover_text(t.hover())
                                    .clicked()
                                    && layer.target != t
                                {
                                    layer.target = t;
                                    // Bipolar parks at -weight in silence, which
                                    // pins a fresh gate shut. Springy reads far
                                    // better as a default for reactions.
                                    if t == LayerTarget::Reaction
                                        && layer.swing == SwingMode::Bipolar
                                    {
                                        layer.swing = SwingMode::Springy;
                                    }
                                    layer.reset_state();
                                }
                            }

                            e.separator();

                            if layer.target == LayerTarget::Force {
                                e.label(RichText::new("combine").weak().small());
                                for m in CombineMode::ALL {
                                    if e
                                        .selectable_label(layer.combine == m, m.label())
                                        .on_hover_text(m.hover())
                                        .clicked()
                                    {
                                        layer.combine = m;
                                    }
                                }
                                e.separator();
                            }

                            e.label(RichText::new("swing").weak().small());
                            for m in SwingMode::ALL {
                                if e
                                    .selectable_label(layer.swing == m, m.label())
                                    .on_hover_text(m.hover())
                                    .clicked()
                                {
                                    layer.swing = m;
                                    layer.reset_state();
                                }
                            }
                        });

                        // ── Reaction gate controls ──────────────────────────
                        if audio.layers[li].target == LayerTarget::Reaction {
                            e.horizontal(|e| {
                                let layer = &mut audio.layers[li];
                                e.add(
                                    egui::Slider::new(&mut layer.threshold, -1.0..=1.0)
                                        .fixed_decimals(2)
                                        .text("gate at"),
                                )
                                .on_hover_text(
                                    "Drive must reach this for the pair's reaction to fire.\n\
                                     Positive weight opens on loud, negative on quiet.",
                                );
                                e.checkbox(&mut layer.drive_rate, "drive rate")
                                    .on_hover_text(
                                        "Also push the global reaction probability\n\
                                         with this layer's peak drive.",
                                    );
                            });

                            if !sim.params.reactions_enabled {
                                e.colored_label(
                                    Color32::from_rgb(210, 180, 90),
                                    "Reactions are off — enable them in the Reactions tab.",
                                );
                            }
                        }

                        // ── Mode-specific params ────────────────────────────
                        {
                            let layer = &mut audio.layers[li];
                            match layer.swing {
                                SwingMode::PingPong => {
                                    e.add(
                                        egui::Slider::new(&mut layer.rate, 0.05..=8.0)
                                            .logarithmic(true)
                                            .text("swings/s at full level"),
                                    );
                                }
                                SwingMode::Springy => {
                                    e.horizontal(|e| {
                                        e.add(
                                            egui::Slider::new(&mut layer.stiffness, 4.0..=400.0)
                                                .logarithmic(true)
                                                .text("stiffness"),
                                        );
                                        e.add(
                                            egui::Slider::new(&mut layer.damping, 0.02..=1.5)
                                                .text("damping"),
                                        )
                                        .on_hover_text("Below ~1.0 rings; above settles flat");
                                    });
                                }
                                SwingMode::Bipolar => {}
                            }
                        }

                        e.label(
                            RichText::new(
                                "drag ↕ adjust · scroll fine · double-click zero · \
                                 right-click flip sign",
                            )
                            .weak()
                            .small(),
                        );

                        // ── The layer's matrix ──────────────────────────────
                        let show_text = cell >= 28.0;
                        Grid::new(("audio_layer_grid", li))
                            .spacing([2.0, 2.0])
                            .show(e, |e| {
                                e.label("");
                                for col in 0..n {
                                    type_header(
                                        e,
                                        col,
                                        cell,
                                        hovered_last
                                            .map_or(false, |(l, _, c)| l == li && c == col),
                                    );
                                }
                                e.end_row();

                                for row in 0..n {
                                    type_header(
                                        e,
                                        row,
                                        cell,
                                        hovered_last
                                            .map_or(false, |(l, r, _)| l == li && r == row),
                                    );

                                    for col in 0..n {
                                        let idx = row * n + col;
                                        let val = audio.layers[li].get(row, col);
                                        let is_rx =
                                            audio.layers[li].target == LayerTarget::Reaction;

                                        // Cells breathe with the live signal so
                                        // you can watch the layer working.
                                        let live = audio.layers[li].drive_at(row, col);
                                        let gate = audio
                                            .outputs
                                            .gate
                                            .get(idx)
                                            .copied()
                                            .unwrap_or(GateState::Untouched);

                                        let color = if is_rx {
                                            gate_color(val, gate, tint)
                                        } else {
                                            weight_color(val, tint, val.abs() * level)
                                        };

                                        let text = show_text.then(|| {
                                            if val == 0.0 {
                                                "·".to_string()
                                            } else if is_rx {
                                                match gate {
                                                    GateState::Open => "▲".to_string(),
                                                    _ => format!("{val:+.2}"),
                                                }
                                            } else {
                                                format!("{val:+.2}")
                                            }
                                        });

                                        let resp = matrix_cell(
                                            e,
                                            cell,
                                            color,
                                            text,
                                            egui::Sense::click_and_drag(),
                                            false,
                                        );

                                        let mut new_val = val;
                                        let mut changed = false;

                                        if resp.dragged() {
                                            new_val = (new_val
                                                - resp.drag_delta().y * 0.005)
                                                .clamp(-1.0, 1.0);
                                            changed = new_val != val;
                                            e.output_mut(|o| {
                                                o.cursor_icon = egui::CursorIcon::ResizeVertical
                                            });
                                        }
                                        if resp.double_clicked() {
                                            new_val = 0.0;
                                            changed = val != 0.0;
                                        }
                                        if resp.secondary_clicked() {
                                            new_val = -val;
                                            changed = val != 0.0;
                                        }
                                        if resp.hovered() {
                                            hovered_now = Some((li, row, col));
                                            let scroll = take_scroll(ctx);
                                            if scroll != 0.0 {
                                                new_val = (new_val + scroll * 0.002)
                                                    .clamp(-1.0, 1.0);
                                                changed = true;
                                            }
                                        }

                                        if changed {
                                            audio.layers[li].set(row, col, new_val);
                                        }

                                        let tip = if is_rx {
                                            let rx = audio.base_reaction(row, col);
                                            let product = if rx < 0 {
                                                "no reaction authored".to_string()
                                            } else {
                                                format!("→ T{rx}")
                                            };
                                            let state = match gate {
                                                GateState::Open => "OPEN",
                                                GateState::Closed => "closed",
                                                GateState::Untouched => "not gated",
                                            };
                                            format!(
                                                "{}  T{row} + T{col} {product}\n\
                                                 weight {val:+.3} · drive {live:+.3} · {state}",
                                                BAND_NAMES[band],
                                            )
                                        } else {
                                            format!(
                                                "{}  T{row} → T{col}\nweight {val:+.3} · \
                                                 drive {live:+.3}",
                                                BAND_NAMES[band],
                                            )
                                        };
                                        resp.on_hover_text(tip);
                                    }
                                    e.end_row();
                                }
                            });
                    });
            }
        });

    audio.hovered = hovered_now;

    if let Some(i) = remove {
        let band = audio.layers[i].band;
        audio.layers.remove(i);
        *status = Some(format!("Removed {} channel", BAND_NAMES[band]));
    }

    if audio.armed && audio.layers.iter().all(|l| !l.enabled || l.active_cells() == 0) {
        e.colored_label(
            Color32::from_rgb(210, 180, 90),
            "Armed but every channel is empty or disabled — nothing will move.",
        );
    }
}

// ── Small painters ──────────────────────────────────────────────────────────

fn level_bar(e: &mut egui::Ui, v: f32, tint: Color32) {
    let (rect, resp) = e.allocate_exact_size(egui::vec2(56.0, 10.0), egui::Sense::hover());
    if e.is_rect_visible(rect) {
        let p = e.painter();
        p.rect_filled(rect, 2.0, Color32::from_gray(38));
        let w = rect.width() * v.clamp(0.0, 1.0);
        p.rect_filled(
            egui::Rect::from_min_size(rect.min, egui::vec2(w, rect.height())),
            2.0,
            tint,
        );
    }
    resp.on_hover_text(format!("live level {v:.2}"));
}

/// Green/red like the force matrix, for the read-only base view.
fn value_color(v: f32) -> Color32 {
    if v > 0.0 {
        Color32::from_rgb(30, (40.0 + v * 180.0) as u8, 30)
    } else if v < 0.0 {
        Color32::from_rgb((40.0 + v.abs() * 180.0) as u8, 30, 30)
    } else {
        Color32::from_rgb(40, 40, 40)
    }
}

/// Reaction cells read as gate state, not magnitude. Open is the band hue at
/// full brightness; closed is the same hue heavily dimmed, so you can still see
/// which cells are wired without them competing with the ones actually firing.
fn gate_color(w: f32, gate: GateState, tint: Color32) -> Color32 {
    if w == 0.0 {
        return Color32::from_rgb(38, 38, 42);
    }
    let k = match gate {
        GateState::Open => 1.0,
        GateState::Closed => 0.30,
        GateState::Untouched => 0.45,
    };
    let mix = |c: u8| (34.0 + (c as f32 - 34.0) * k) as u8;
    let out = Color32::from_rgb(mix(tint.r()), mix(tint.g()), mix(tint.b()));
    if w < 0.0 {
        // Inverted cells (open on quiet) get a cooler read.
        Color32::from_rgb(
            (out.r() as f32 * 0.55) as u8,
            (out.g() as f32 * 0.70) as u8,
            out.b(),
        )
    } else {
        out
    }
}

/// Layer cells are tinted with the band hue, brightened by the live signal.
/// Sign is carried by luminance direction so negatives still read as "pulling
/// the other way" without a second hue fighting the band identity.
fn weight_color(w: f32, tint: Color32, lit: f32) -> Color32 {
    if w == 0.0 {
        return Color32::from_rgb(38, 38, 42);
    }
    let mag = w.abs().clamp(0.0, 1.0);
    let base = if w > 0.0 { 0.35 } else { 0.15 };
    let k = (base + 0.55 * mag + 0.35 * lit).clamp(0.0, 1.0);
    let mix = |c: u8| (34.0 + (c as f32 - 34.0) * k) as u8;
    let out = Color32::from_rgb(mix(tint.r()), mix(tint.g()), mix(tint.b()));
    if w < 0.0 {
        // Negative weights get a cooler, desaturated read.
        Color32::from_rgb(
            (out.r() as f32 * 0.55) as u8,
            (out.g() as f32 * 0.70) as u8,
            out.b(),
        )
    } else {
        out
    }
}

fn fmt_time(t: f32) -> String {
    let t = t.max(0.0) as u32;
    format!("{}:{:02}", t / 60, t % 60)
}
