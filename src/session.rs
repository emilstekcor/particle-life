use crate::{
    sim::{Particle, SimParams, SimState, MAX_CPU_PHYSICS_PARTICLES, MAX_TYPES},
    ui::UiState,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, time::Instant};
#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    version: u32,
    pub params: SimParams,
    pub particles: Vec<Particle>,
    pub force_matrix: Vec<f32>,
    pub reaction_table: Vec<i32>,
    pub trace_len_matrix: Vec<u32>,
    pub step: u64,
    next_prefab_id: i32,
    gpu: bool,
    seed: u64,
    random_counter: u64,
    view: crate::multidim::ViewNd,
    centers: [f32; 8],
    widths: [f32; 8],
    camera: [f32; 5],
    playback: Playback,
    strobe: bool,
    trace_len: u32,
    trace_mode: crate::ui::TraceRenderMode,
    trace_filter: i32,
    trace_trigger: bool,
    trace_fade: f32,
    audio: crate::audio::AudioSnapshot,
}
impl Snapshot {
    pub fn capture(sim: &SimState, ui: &UiState) -> Self {
        Self {
            version: 1,
            params: sim.params.clone(),
            particles: sim.particles.clone(),
            force_matrix: sim.force_matrix.clone(),
            reaction_table: sim.reaction_table.clone(),
            trace_len_matrix: sim.trace_len_matrix.clone(),
            step: sim.step_count,
            next_prefab_id: sim.next_prefab_instance_id,
            gpu: ui.use_gpu_physics,
            seed: sim.seed,
            random_counter: sim.random_counter,
            view: ui.nd.clone(),
            centers: ui.extra_slice_centers,
            widths: ui.extra_slice_thickness,
            camera: [
                ui.fly_pos.x,
                ui.fly_pos.y,
                ui.fly_pos.z,
                ui.fly_yaw,
                ui.fly_pitch,
            ],
            playback: ui.playback.clone(),
            strobe: ui.strobe,
            trace_len: ui.trace_len,
            trace_mode: ui.trace_render_mode,
            trace_filter: ui.trace_type_filter,
            trace_trigger: ui.trace_trigger_only,
            trace_fade: ui.trace_fade_alpha,
            audio: ui.audio.snapshot(),
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let p = &self.params;
        let n = p.type_count;
        if self.version != 1 {
            return Err("Unsupported session version".into());
        }
        if !(3..=8).contains(&p.dimension)
            || !(1..=MAX_TYPES).contains(&n)
            || self.particles.len() > MAX_CPU_PHYSICS_PARTICLES
        {
            return Err("Invalid dimension, type or particle count".into());
        }
        if !p.bounds.is_finite()
            || p.bounds <= 0.0
            || !p.dt.is_finite()
            || p.dt <= 0.0
            || !p.beta.is_finite()
            || p.beta <= 0.0
            || p.beta >= 1.0
            || !(0.0..=1.0).contains(&p.friction)
            || !(0.0..=1.0).contains(&p.reaction_probability)
        {
            return Err("Invalid physics parameters".into());
        }
        for x in [
            p.r_max,
            p.force_scale,
            p.max_speed,
            p.mix_radius,
            p.particle_size,
        ] {
            if !x.is_finite() || x < 0.0 {
                return Err("Invalid physics magnitude".into());
            }
        }
        if self.force_matrix.len() != n * n
            || self.reaction_table.len() != n * n
            || self.trace_len_matrix.len() != n * n
            || self.force_matrix.iter().any(|x| !x.is_finite())
            || self.reaction_table.iter().any(|&x| x < -1 || x >= n as i32)
        {
            return Err("Invalid rule matrices".into());
        }
        if self.particles.iter().any(|p| {
            p.kind >= n as u32
                || p.position
                    .iter()
                    .chain(p.velocity.iter())
                    .any(|x| !x.is_finite())
        }) {
            return Err("Invalid particle state".into());
        }
        if self
            .view
            .angles
            .iter()
            .flatten()
            .chain(self.centers.iter())
            .chain(self.widths.iter())
            .chain(self.camera.iter())
            .any(|v| !v.is_finite())
            || self.widths.iter().any(|&x| x <= 0.0)
        {
            return Err("Invalid view state".into());
        }
        self.audio.validate(n)?;
        Ok(())
    }
    pub fn restore(self, sim: &mut SimState, ui: &mut UiState) {
        sim.params = self.params;
        sim.particles = self.particles;
        sim.force_matrix = self.force_matrix;
        sim.reaction_table = self.reaction_table;
        sim.trace_len_matrix = self.trace_len_matrix;
        sim.step_count = self.step;
        sim.next_prefab_instance_id = self.next_prefab_id;
        sim.seed = self.seed;
        sim.random_counter = self.random_counter;
        sim.trace_timers = vec![0; sim.particles.len()];
        sim.particles_dirty = true;
        sim.particles_replaced = true;
        sim.params_dirty = true;
        sim.force_matrix_dirty = true;
        sim.reaction_table_dirty = true;
        sim.trace_len_matrix_dirty = true;
        ui.use_gpu_physics = self.gpu;
        ui.nd = self.view;
        ui.nd.normalize(sim.params.dimension);
        ui.extra_slice_centers = self.centers;
        ui.extra_slice_thickness = self.widths;
        ui.fly_pos = glam::Vec3::new(self.camera[0], self.camera[1], self.camera[2]);
        ui.fly_yaw = self.camera[3];
        ui.fly_pitch = self.camera[4];
        ui.playback = self.playback;
        ui.playback.reset();
        ui.strobe = self.strobe;
        ui.trace_len = self.trace_len;
        ui.trace_render_mode = self.trace_mode;
        ui.trace_type_filter = self.trace_filter;
        ui.trace_trigger_only = self.trace_trigger;
        ui.trace_fade_alpha = self.trace_fade;
        ui.audio.restore_snapshot(self.audio);
        ui.paused = true;
        ui.step_once = false;
        ui.pending_dimension = None;
        ui.selected_indices.clear();
        ui.clear_selection_requested = true;
        ui.type_mix.clear();
    }
}
/// Write a synced temporary file, keep a backup, then rename in the same directory.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let temp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&temp).map_err(|e| e.to_string())?;
    f.write_all(bytes)
        .and_then(|_| f.sync_all())
        .map_err(|e| e.to_string())?;
    if path.exists() {
        std::fs::copy(path, path.with_extension("bak")).map_err(|e| e.to_string())?;
    }
    std::fs::rename(&temp, path).map_err(|e| e.to_string())
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Playback {
    pub realtime: bool,
    pub steps_per_second: f64,
    pub display_every: u32,
    #[serde(skip)]
    last: Option<Instant>,
    #[serde(skip)]
    accumulator: f64,
    #[serde(skip)]
    pub behind: bool,
}
impl Default for Playback {
    fn default() -> Self {
        Self {
            realtime: false,
            steps_per_second: 60.0,
            display_every: 1,
            last: None,
            accumulator: 0.0,
            behind: false,
        }
    }
}
impl Playback {
    pub fn reset(&mut self) {
        self.last = None;
        self.accumulator = 0.0;
        self.behind = false;
    }
    pub fn steps(&mut self, paused: bool, single: bool, strobe: bool) -> u32 {
        let now = Instant::now();
        let elapsed = self
            .last
            .replace(now)
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);
        self.advance(elapsed, paused, single, strobe)
    }
    fn advance(&mut self, elapsed: f64, paused: bool, single: bool, strobe: bool) -> u32 {
        self.behind = false;
        if paused {
            self.accumulator = 0.0;
            return u32::from(single);
        }
        let stride = if strobe {
            2
        } else {
            self.display_every.clamp(1, 16)
        };
        if !self.realtime {
            self.accumulator = 0.0;
            return stride;
        }
        self.accumulator += elapsed.min(0.25) * self.steps_per_second.clamp(1.0, 1000.0);
        let due = (self.accumulator / stride as f64).floor() as u32 * stride;
        let cap = (16 / stride).max(1) * stride;
        let steps = due.min(cap);
        self.accumulator -= steps as f64;
        if due > cap {
            self.behind = true;
            self.accumulator = self.accumulator.min(stride as f64);
        }
        steps
    }
}
pub fn playback_controls(e: &mut egui::Ui, p: &mut Playback) {
    if e.checkbox(&mut p.realtime, "Fixed steps / second")
        .changed()
    {
        p.reset();
    }
    if p.realtime {
        e.add(egui::Slider::new(&mut p.steps_per_second, 1.0..=480.0).text("steps / second"));
    }
    e.add(egui::Slider::new(&mut p.display_every, 1..=16).text("display stride"));
    if p.behind {
        e.colored_label(
            egui::Color32::YELLOW,
            "Simulation cannot keep up; reduce workload or playback rate.",
        );
    }
}
#[derive(Default)]
pub struct SessionUi {
    pub open: bool,
    pub save_requested: bool,
    pub load_requested: bool,
    pub undo_requested: bool,
    pub redo_requested: bool,
    pub inspect_requested: bool,
    pub capture_slot: Option<usize>,
    pub restore_slot: Option<usize>,
    experiments: [Option<Snapshot>; 2],
    pub restore_prefab_rules: bool,
    pub save_live_creature: bool,
    undo: Vec<Snapshot>,
    redo: Vec<Snapshot>,
    inspection: Option<Inspection>,
}
impl SessionUi {
    pub fn new() -> Self {
        Self {
            save_live_creature: true,
            ..Default::default()
        }
    }
    pub fn needs_live_state(&self) -> bool {
        self.save_requested
            || self.load_requested
            || self.undo_requested
            || self.redo_requested
            || self.inspect_requested
            || self.capture_slot.is_some()
            || self.restore_slot.is_some()
    }
    pub fn push_undo(&mut self, s: Snapshot) {
        if self.undo.len() >= 8 {
            self.undo.remove(0);
        }
        self.undo.push(s);
        self.redo.clear();
    }
}
struct Inspection {
    step: u64,
    count: usize,
    counts: Vec<usize>,
    speed: f32,
    spread: f32,
    center: [f32; 8],
    visible: usize,
}
pub fn wrapped_delta(d: f32, b: f32) -> f32 {
    (d + b * 0.5).rem_euclid(b) - b * 0.5
}
fn inspect(sim: &SimState, ui: &UiState) -> Inspection {
    let particles: Vec<_> = if ui.selected_indices.is_empty() {
        sim.particles.iter().collect()
    } else {
        ui.selected_indices
            .iter()
            .filter_map(|&i| sim.particles.get(i))
            .collect()
    };
    let mut center = [0.0; 8];
    let mut velocity = [0.0; 8];
    let mut counts = vec![0; sim.params.type_count];
    let reference = particles.first().map(|p| p.position).unwrap_or([0.0; 8]);
    for p in &particles {
        counts[p.kind as usize] += 1;
        for d in 0..sim.params.dimension {
            let delta = p.position[d] - reference[d];
            center[d] += reference[d]
                + if sim.params.wrap {
                    wrapped_delta(delta, sim.params.bounds)
                } else {
                    delta
                };
            velocity[d] += p.velocity[d];
        }
    }
    let n = particles.len().max(1) as f32;
    for d in 0..sim.params.dimension {
        center[d] /= n;
        velocity[d] /= n;
    }
    let spread = (particles
        .iter()
        .map(|p| {
            (0..sim.params.dimension)
                .map(|d| {
                    let delta = p.position[d] - center[d];
                    let delta = if sim.params.wrap {
                        wrapped_delta(delta, sim.params.bounds)
                    } else {
                        delta
                    };
                    delta * delta
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        / n)
        .sqrt();
    let speed = velocity.iter().map(|v| v * v).sum::<f32>().sqrt();
    let visible = particles
        .iter()
        .filter(|p| {
            let q = ui
                .nd
                .project(p.position, sim.params.dimension, sim.params.bounds);
            (3..sim.params.dimension).all(|d| {
                (q[d] - ui.extra_slice_centers[d]).abs() < ui.extra_slice_thickness[d] * 0.5
            })
        })
        .count();
    Inspection {
        step: sim.step_count,
        count: particles.len(),
        counts,
        speed,
        spread,
        center,
        visible,
    }
}
fn compare(a: &Snapshot, b: &Snapshot) -> Option<(f32, f32)> {
    if a.params.dimension != b.params.dimension
        || a.params.bounds != b.params.bounds
        || a.params.wrap != b.params.wrap
        || a.particles.len() != b.particles.len()
        || a.particles.is_empty()
    {
        return None;
    }
    let dim = a.params.dimension;
    let n = a.particles.len() as f32;
    let mut drift = [0.0; 8];
    let difference = |x: f32| {
        if a.params.wrap {
            wrapped_delta(x, a.params.bounds)
        } else {
            x
        }
    };
    for (x, y) in a.particles.iter().zip(&b.particles) {
        for d in 0..dim {
            drift[d] += difference(y.position[d] - x.position[d]) / n;
        }
    }
    let mut position = 0.0;
    let mut velocity = 0.0;
    for (x, y) in a.particles.iter().zip(&b.particles) {
        for d in 0..dim {
            position += difference(y.position[d] - x.position[d] - drift[d]).powi(2);
            velocity += (y.velocity[d] - x.velocity[d]).powi(2);
        }
    }
    Some(((position / n).sqrt(), (velocity / n).sqrt()))
}
pub fn draw(ctx: &egui::Context, sim: &mut SimState, ui: &mut UiState) {
    if !ui.session.open {
        return;
    }
    let mut open = true;
    egui::Window::new("Session & creature inspector").open(&mut open).default_pos([875.0,100.0]).default_width(330.0).vscroll(true).show(ctx,|e|{
  e.horizontal(|e|{if e.button("Save session…").clicked(){ui.session.save_requested=true;}if e.button("Load session…").clicked(){ui.session.load_requested=true;}});e.small("Sessions restore paused. Audio settings and file references are included; trail history restarts.");
  e.horizontal(|e|{if e.add_enabled(!ui.session.undo.is_empty(),egui::Button::new("Undo edit")).clicked(){ui.session.undo_requested=true;}if e.add_enabled(!ui.session.redo.is_empty(),egui::Button::new("Redo edit")).clicked(){ui.session.redo_requested=true;}});
  e.label("Random generation seed");e.add(egui::DragValue::new(&mut sim.seed));if e.button("Reset random sequence").clicked(){sim.random_counter=0;ui.flash("Random sequence reset; next spawn/randomize uses this seed");}
  e.separator();if e.button("Inspect selection (or whole world)").clicked(){ui.session.inspect_requested=true;}
  if let Some(i)=&ui.session.inspection{e.label(format!("Sample: step {} • {} particles",i.step,i.count));e.label(format!("Inside hidden-axis slices: {}",i.visible));e.label(format!("Center speed: {:.6}\nRMS spread: {:.6}",i.speed,i.spread));for d in 0..sim.params.dimension{e.label(format!("{} center: {:.5}",crate::multidim::AXES[d],i.center[d]));}for(kind,count)in i.counts.iter().enumerate(){e.label(format!("Type {kind}: {count}"));}}
  e.separator();e.label("A/B experiment checkpoints");for slot in 0..2{e.horizontal(|e|{if e.button(format!("Capture {}",["A","B"][slot])).clicked(){ui.session.capture_slot=Some(slot);}if e.add_enabled(ui.session.experiments[slot].is_some(),egui::Button::new(format!("Restore {}",["A","B"][slot]))).clicked(){ui.session.restore_slot=Some(slot);}});}
  if let[Some(a),Some(b)]=&ui.session.experiments{e.label(format!("A: step {}, {} particles • B: step {}, {} particles",a.step,a.particles.len(),b.step,b.particles.len()));if let Some((position,velocity))=compare(a,b){e.label(format!("Translation-aligned RMS: position {position:.7}, velocity {velocity:.7}"));e.small("A small residual suggests a return after the step difference; it does not establish the shortest period. Compare the same particle ordering.");}}
  e.small("A/B slots last for this run; save a session to keep an experiment.");e.separator();let n=sim.particles.len() as u64;e.label(format!("All-pairs upper bound: {} checks / force step",n.saturating_mul(n.saturating_sub(1))));if ui.use_gpu_physics&&n>10_000&&sim.params.reactions_enabled&&crate::renderer::compute::GpuParams::from(&sim.params).grid_res==0{e.colored_label(egui::Color32::YELLOW,"GPU reactions suspended above 10,000 particles");}
 });
    ui.session.open = open;
}
pub fn process(sim: &mut SimState, ui: &mut UiState) {
    if let Some(slot) = ui.session.capture_slot.take() {
        ui.session.experiments[slot] = Some(Snapshot::capture(sim, ui));
        ui.flash("Experiment captured");
    }
    if let Some(slot) = ui.session.restore_slot.take() {
        if let Some(s) = ui.session.experiments[slot].clone() {
            let before = Snapshot::capture(sim, ui);
            ui.session.push_undo(before);
            s.restore(sim, ui);
            ui.flash("Experiment restored; paused");
        }
    }
    if std::mem::take(&mut ui.session.undo_requested) {
        if let Some(previous) = ui.session.undo.pop() {
            let now = Snapshot::capture(sim, ui);
            ui.session.redo.push(now);
            previous.restore(sim, ui);
            ui.flash("Edit undone; paused");
        }
    }
    if std::mem::take(&mut ui.session.redo_requested) {
        if let Some(next) = ui.session.redo.pop() {
            let now = Snapshot::capture(sim, ui);
            ui.session.undo.push(now);
            next.restore(sim, ui);
            ui.flash("Edit restored; paused");
        }
    }
    if std::mem::take(&mut ui.session.inspect_requested) {
        ui.session.inspection = Some(inspect(sim, ui));
    }
    if std::mem::take(&mut ui.session.save_requested) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Particle Life session", &["json"])
            .set_file_name("particle-life-session.json")
            .save_file()
        {
            let result = serde_json::to_vec(&Snapshot::capture(sim, ui))
                .map_err(|e| e.to_string())
                .and_then(|bytes| atomic_write(&path, &bytes));
            ui.flash(match result {
                Ok(()) => "Session saved".into(),
                Err(e) => format!("Save failed: {e}"),
            });
        }
    }
    if std::mem::take(&mut ui.session.load_requested) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Particle Life session", &["json"])
            .pick_file()
        {
            let result = (|| {
                if std::fs::metadata(&path).map_err(|e| e.to_string())?.len() > 128 * 1024 * 1024 {
                    return Err("Session exceeds 128 MiB".into());
                }
                let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
                let s: Snapshot = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                s.validate()?;
                Ok::<_, String>(s)
            })();
            match result {
                Ok(s) => {
                    let before = Snapshot::capture(sim, ui);
                    ui.session.push_undo(before);
                    s.restore(sim, ui);
                    ui.flash("Session loaded; paused");
                }
                Err(e) => ui.flash(format!("Load failed: {e}")),
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn playback_is_independent_of_frame_rate() {
        for hz in [30, 60, 144] {
            let mut p = Playback::default();
            p.realtime = true;
            p.steps_per_second = 120.0;
            let total: u32 = (0..hz)
                .map(|_| p.advance(1.0 / hz as f64, false, false, false))
                .sum();
            assert!((total as i32 - 120).abs() <= 1);
        }
    }
    #[test]
    fn paused_single_step_and_strobe() {
        let mut p = Playback::default();
        assert_eq!(p.advance(1.0, true, true, true), 1);
        assert_eq!(p.advance(1.0, true, false, true), 0);
        assert_eq!(p.advance(1.0, false, false, true), 2);
    }
    #[test]
    fn session_roundtrip() {
        let sim = SimState::new();
        let ui = UiState::new();
        let s = Snapshot::capture(&sim, &ui);
        let bytes = serde_json::to_vec(&s).unwrap();
        let copy: Snapshot = serde_json::from_slice(&bytes).unwrap();
        copy.validate().unwrap();
        assert_eq!(copy.particles[0].position, sim.particles[0].position);
    }
    #[test]
    fn invalid_session_rejected() {
        let sim = SimState::new();
        let ui = UiState::new();
        let mut s = Snapshot::capture(&sim, &ui);
        s.reaction_table[0] = 999;
        assert!(s.validate().is_err());
    }
    #[test]
    fn checkpoint_restores_deleted_and_moved_particles() {
        let mut sim = SimState::new();
        let mut ui = UiState::new();
        sim.particles[2].velocity[0] = 0.75;
        let original = sim.particles.clone();
        let s = Snapshot::capture(&sim, &ui);
        sim.delete_particles(&[1]);
        sim.move_particles(&[0], glam::Vec3::splat(0.25));
        sim.scale_velocities(0.0);
        s.restore(&mut sim, &mut ui);
        assert_eq!(sim.particles.len(), original.len());
        for (a, b) in sim.particles.iter().zip(original) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.velocity, b.velocity);
            assert_eq!(a.kind, b.kind);
        }
        assert!(sim.particles_replaced && sim.particles_dirty && ui.paused);
    }
    #[test]
    fn reset_seed_reproduces_spawn_and_rules() {
        let mut sim = SimState::new();
        sim.seed = 452;
        sim.random_counter = 0;
        sim.randomize_rules();
        let matrix = sim.force_matrix.clone();
        sim.respawn_mix(&[4, 4, 4, 4]);
        let first = sim.particles.clone();
        sim.random_counter = 0;
        sim.randomize_rules();
        sim.respawn_mix(&[4, 4, 4, 4]);
        assert_eq!(sim.force_matrix, matrix);
        for (a, b) in sim.particles.iter().zip(first) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.kind, b.kind);
        }
    }
}
