//! Audio modulation panel — lives inside the Rules tab of the Matrix Editor.

use crate::audio::{AudioMod, Route, BANDS, BAND_NAMES};
use crate::sim::SimState;
use egui::{Color32, RichText};

/// Returns an optional status message for the caller to flash.
pub fn draw_audio_mod_ui(e: &mut egui::Ui, sim: &mut SimState, audio: &mut AudioMod) -> Option<String> {
    let mut status: Option<String> = None;
    let n = sim.params.type_count;

    egui::CollapsingHeader::new("🎵 Audio modulation")
        .default_open(false)
        .show(e, |e| {
            // ── File select ─────────────────────────────────────────────────
            e.horizontal(|e| {
                if e.button("Load audio…").clicked() {
                    if audio.pick_file() {
                        status = Some(format!("Loaded {}", audio.file_name));
                    } else if let Some(err) = &audio.load_error {
                        status = Some(err.clone());
                    }
                }
                if audio.has_file() {
                    e.label(RichText::new(&audio.file_name).strong());
                    if e.button("✖").on_hover_text("Unload").clicked() {
                        audio.disarm(sim);
                        audio.stop();
                        *audio = AudioMod::new();
                        status = Some("Audio unloaded".into());
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
                return;
            }

            // ── Transport ───────────────────────────────────────────────────
            e.horizontal(|e| {
                let label = if audio.playing { "⏸ Pause" } else { "▶ Play" };
                if e.button(label).clicked() {
                    audio.toggle_play();
                }
                if e.button("⏹ Stop").clicked() {
                    audio.stop();
                }
                e.checkbox(&mut audio.looping, "loop");

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

            e.separator();

            // ── Arm + global shape ──────────────────────────────────────────
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
                        status = Some("Matrix snapshot captured".into());
                    } else {
                        audio.disarm(sim);
                        status = Some("Matrix restored".into());
                    }
                }

                if audio.armed && e.button("Re-snapshot").on_hover_text(
                    "Take a new base from the current matrix",
                ).clicked() {
                    audio.capture_base(sim);
                    status = Some("New base captured".into());
                }
            });

            e.add(
                egui::Slider::new(&mut audio.depth, 0.0..=1.0)
                    .text("depth")
                    .fixed_decimals(2),
            )
            .on_hover_text("How far cells swing from their base value. Start small.");

            e.add(
                egui::Slider::new(&mut audio.smoothing, 0.0..=0.98)
                    .text("smoothing")
                    .fixed_decimals(2),
            )
            .on_hover_text("Slew limiting. 0 is raw and jittery, ~0.85 glides.");

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

            e.separator();

            // ── Band meters ─────────────────────────────────────────────────
            e.label(RichText::new("bands").weak());
            e.horizontal(|e| {
                for b in 0..BANDS {
                    draw_band_meter(e, audio.bands[b], BAND_NAMES[b], b);
                }
            });

            e.separator();

            // ── Routes ──────────────────────────────────────────────────────
            e.horizontal(|e| {
                e.label(RichText::new("routes").weak());
                if e.button("+ add").clicked() {
                    audio.routes.push(Route::new(0, 0, 0));
                }
                if e.button("auto-map").on_hover_text(
                    "One band per off-diagonal cell, alternating sign",
                ).clicked() {
                    audio.auto_map(n, true);
                    status = Some(format!("{} routes mapped", audio.routes.len()));
                }
                if e.button("clear").clicked() {
                    audio.routes.clear();
                }
            });

            let mut remove: Option<usize> = None;
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .id_source("audio_routes")
                .show(e, |e| {
                    egui::Grid::new("route_grid")
                        .num_columns(7)
                        .spacing([6.0, 4.0])
                        .show(e, |e| {
                            for (idx, r) in audio.routes.iter_mut().enumerate() {
                                e.checkbox(&mut r.enabled, "");

                                e.add(
                                    egui::DragValue::new(&mut r.row)
                                        .clamp_range(0..=n.saturating_sub(1))
                                        .prefix("T"),
                                );
                                e.label("→");
                                e.add(
                                    egui::DragValue::new(&mut r.col)
                                        .clamp_range(0..=n.saturating_sub(1))
                                        .prefix("T"),
                                );

                                egui::ComboBox::from_id_source(("band", idx))
                                    .selected_text(BAND_NAMES[r.band.min(BANDS - 1)])
                                    .width(90.0)
                                    .show_ui(e, |e| {
                                        for b in 0..BANDS {
                                            e.selectable_value(&mut r.band, b, BAND_NAMES[b]);
                                        }
                                    });

                                e.add(
                                    egui::DragValue::new(&mut r.gain)
                                        .speed(0.02)
                                        .clamp_range(-2.0..=2.0)
                                        .fixed_decimals(2)
                                        .prefix("×"),
                                );

                                e.horizontal(|e| {
                                    e.checkbox(&mut r.bipolar, "±")
                                        .on_hover_text("Swing below the base value too");
                                    if e.button("🗑").clicked() {
                                        remove = Some(idx);
                                    }
                                });
                                e.end_row();
                            }
                        });
                });

            if let Some(i) = remove {
                audio.routes.remove(i);
            }

            if audio.armed && audio.routes.is_empty() {
                e.colored_label(
                    Color32::from_rgb(210, 180, 90),
                    "Armed but no routes — nothing will move.",
                );
            }
        });

    status
}

fn draw_band_meter(e: &mut egui::Ui, v: f32, name: &str, idx: usize) {
    let w = 26.0;
    let h = 54.0;
    let (rect, resp) = e.allocate_exact_size(egui::vec2(w, h), egui::Sense::hover());
    if e.is_rect_visible(rect) {
        let p = e.painter();
        p.rect_filled(rect, 3.0, Color32::from_gray(38));
        let fill_h = h * v.clamp(0.0, 1.0);
        let fill_rect = egui::Rect::from_min_max(
            egui::pos2(rect.min.x, rect.max.y - fill_h),
            rect.max,
        );
        // hue walks across the spectrum so bands are visually distinguishable
        let t = idx as f32 / (BANDS - 1).max(1) as f32;
        let color = Color32::from_rgb(
            (60.0 + 190.0 * t) as u8,
            (200.0 - 90.0 * t) as u8,
            (220.0 - 140.0 * t) as u8,
        );
        p.rect_filled(fill_rect, 3.0, color);
    }
    resp.on_hover_text(format!("{name}: {v:.2}"));
}

fn fmt_time(t: f32) -> String {
    let t = t.max(0.0) as u32;
    format!("{}:{:02}", t / 60, t % 60)
}
