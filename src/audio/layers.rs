//! Band layers — the matrix-shaped replacement for the old flat `Route` list.
//!
//! A layer owns one frequency band and one N×N weight matrix laid out exactly
//! like the force matrix (`row * n + col`, cells clamped to -1..1). A cell of
//! 0.0 means "no connection", which is precisely what an absent route used to
//! mean, so the old model round-trips into this one without loss.
//!
//! Each frame a layer turns its band's 0..1 envelope into a per-cell *drive*
//! value. How that drive is shaped is the swing mode; where it lands is the
//! target; how it lands is the combine mode. All three are exclusive choices
//! rendered as latching push buttons in the UI.
//!
//! ## Why reactions work differently
//!
//! The force matrix is continuous — a cell is an f32 in -1..1, so a drive
//! value can push it anywhere. The reaction table is not: a cell is an i32
//! naming the type a pair turns into, or -1 for no reaction. There is no
//! "34% of becoming type 2", and `reaction_probability` is a single global
//! scalar shared by every pair. So reaction layers don't offset a value, they
//! **gate** one: above threshold the pair's authored reaction is live, below
//! it the cell reads -1 and the pair passes through inert. Same snapshot,
//! same restore, same swing modes — discrete output instead of continuous.

/// What a layer's matrix drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerTarget {
    /// The force matrix. Continuous — see `CombineMode`.
    Force,
    /// The reaction table. Discrete — see `GateState` and `threshold`.
    Reaction,
}

impl LayerTarget {
    pub fn label(self) -> &'static str {
        match self {
            LayerTarget::Force => "Force",
            LayerTarget::Reaction => "Reaction",
        }
    }

    pub fn hover(self) -> &'static str {
        match self {
            LayerTarget::Force => "Offsets or scales force matrix cells.",
            LayerTarget::Reaction => {
                "Gates reaction table cells on and off.\n\
                 Positive weight opens on loud, negative opens on quiet."
            }
        }
    }

    pub const ALL: [LayerTarget; 2] = [LayerTarget::Force, LayerTarget::Reaction];
}

/// How a Force layer's drive lands on the base matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineMode {
    /// `base + drive * depth`. Cells sitting at zero can come alive.
    Add,
    /// `base * (1 + drive * depth)`. Animates what you authored — a cell at
    /// zero stays at zero, and cells with big authored values swing hardest.
    Scale,
}

impl CombineMode {
    pub fn label(self) -> &'static str {
        match self {
            CombineMode::Add => "Add",
            CombineMode::Scale => "Scale",
        }
    }

    pub fn hover(self) -> &'static str {
        match self {
            CombineMode::Add => "base + drive × depth\nZeroed cells can come alive.",
            CombineMode::Scale => {
                "base × (1 + drive × depth)\nAnimates what you authored; a zeroed cell stays zero."
            }
        }
    }

    pub const ALL: [CombineMode; 2] = [CombineMode::Add, CombineMode::Scale];
}

/// How the band envelope is shaped into a drive value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwingMode {
    /// Envelope maps to -1..1, so the cell rides above *and* below its base.
    /// Silence parks it at -weight, full level at +weight.
    Bipolar,
    /// A triangle oscillator whose amplitude is the band level and whose rate
    /// climbs with it. Loud passages swing wide and fast, quiet ones go still.
    PingPong,
    /// A damped spring chasing the envelope. Overshoots transients and rings
    /// down instead of tracking exactly — gives hits a physical follow-through.
    Springy,
}

impl SwingMode {
    pub fn label(self) -> &'static str {
        match self {
            SwingMode::Bipolar => "Bipolar",
            SwingMode::PingPong => "Ping-pong",
            SwingMode::Springy => "Springy",
        }
    }

    pub fn hover(self) -> &'static str {
        match self {
            SwingMode::Bipolar => {
                "Rides above and below the base value.\nSilence pulls down, level pushes up."
            }
            SwingMode::PingPong => {
                "Triangle oscillator. Amplitude and rate both\nfollow the band — loud swings wide and fast."
            }
            SwingMode::Springy => {
                "Damped spring chasing the band. Overshoots\ntransients and rings down."
            }
        }
    }

    pub const ALL: [SwingMode; 3] = [SwingMode::Bipolar, SwingMode::PingPong, SwingMode::Springy];
}

/// Resolved gate for one reaction cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// No reaction layer weights this cell — it keeps whatever was authored.
    Untouched,
    /// At least one layer is holding it open.
    Open,
    /// Weighted by a layer, but no layer is over threshold right now.
    Closed,
}

/// One band → one weight matrix.
pub struct BandLayer {
    pub band: usize,
    /// `types * types` cells, each clamped to -1..1. Row-major, same layout as
    /// `SimState::force_matrix` and `SimState::reaction_table`.
    pub weights: Vec<f32>,
    /// Matrix dimension the weights were allocated for.
    pub types: usize,

    pub enabled: bool,
    /// Layer-wide multiplier on top of every cell. Negative inverts the layer.
    pub gain: f32,
    pub target: LayerTarget,
    pub combine: CombineMode,
    pub swing: SwingMode,

    /// Reaction target: drive must reach this for the gate to open.
    pub threshold: f32,
    /// Reaction target: also push the global reaction probability.
    pub drive_rate: bool,

    /// Ping-pong: swings per second at full level.
    pub rate: f32,
    /// Springy: how hard the spring pulls toward the target.
    pub stiffness: f32,
    /// Springy: 1.0 is roughly critical, below that rings.
    pub damping: f32,

    // ── Per-frame / integrator state ────────────────────────────────────────
    /// Drive value per cell, refilled every frame by `drive()`.
    out: Vec<f32>,
    /// Ping-pong phase in 0..1.
    phase: f32,
    /// Springy position and velocity per cell.
    spring_x: Vec<f32>,
    spring_v: Vec<f32>,
    /// Collapsed state in the UI.
    pub open: bool,
}

impl BandLayer {
    pub fn new(band: usize, types: usize) -> Self {
        Self {
            band,
            weights: vec![0.0; types * types],
            types,
            enabled: true,
            gain: 1.0,
            target: LayerTarget::Force,
            combine: CombineMode::Add,
            swing: SwingMode::Bipolar,
            threshold: 0.35,
            drive_rate: false,
            rate: 1.5,
            stiffness: 60.0,
            damping: 0.35,
            out: vec![0.0; types * types],
            phase: 0.0,
            spring_x: vec![0.0; types * types],
            spring_v: vec![0.0; types * types],
            open: true,
        }
    }

    /// A reaction layer defaults to Springy rather than Bipolar. Bipolar parks
    /// at `-weight` in silence, which for a positive-weight cell means the gate
    /// is pinned shut until the band is over halfway up — a confusing default
    /// for something whose whole job is to open and close.
    pub fn new_reaction(band: usize, types: usize) -> Self {
        let mut l = Self::new(band, types);
        l.target = LayerTarget::Reaction;
        l.swing = SwingMode::Springy;
        l
    }

    /// Grow or shrink to `n` types, keeping the top-left block. Mirrors what
    /// `SimState::set_type_count` does to the force matrix so a layer never
    /// goes stale against the sim.
    pub fn resize(&mut self, n: usize) {
        if self.types == n && self.weights.len() == n * n {
            return;
        }
        let old = self.types;
        let mut next = vec![0.0f32; n * n];
        for i in 0..old.min(n) {
            for j in 0..old.min(n) {
                if let Some(v) = self.weights.get(i * old + j) {
                    next[i * n + j] = *v;
                }
            }
        }
        self.weights = next;
        self.types = n;
        self.out = vec![0.0; n * n];
        self.spring_x = vec![0.0; n * n];
        self.spring_v = vec![0.0; n * n];
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.weights
            .get(row * self.types + col)
            .copied()
            .unwrap_or(0.0)
    }

    pub fn set(&mut self, row: usize, col: usize, v: f32) {
        let n = self.types;
        if row < n && col < n {
            self.weights[row * n + col] = v.clamp(-1.0, 1.0);
        }
    }

    /// Live drive for one cell — what the UI paints and reports.
    pub fn drive_at(&self, row: usize, col: usize) -> f32 {
        self.out.get(row * self.types + col).copied().unwrap_or(0.0)
    }

    /// Number of non-zero cells — what the old UI would have called routes.
    pub fn active_cells(&self) -> usize {
        self.weights.iter().filter(|w| **w != 0.0).count()
    }

    pub fn clear(&mut self) {
        for w in &mut self.weights {
            *w = 0.0;
        }
        self.reset_state();
    }

    /// Advance this layer's state, filling `out` with a per-cell drive value.
    ///
    /// `level` is the band's 0..1 envelope. `out` is *pre-depth*; the global
    /// depth is applied once at combine time so the depth slider means the same
    /// thing across every layer. Reaction gates compare against this pre-depth
    /// value too, so nudging depth doesn't silently move every threshold in the
    /// rig.
    ///
    /// Returns nothing on purpose — handing back `&[f32]` would keep `self`
    /// mutably borrowed, and `accumulate` needs to read `weights` and `out`
    /// side by side.
    pub fn drive(&mut self, level: f32, dt: f32) {
        let n = self.types;
        if self.out.len() != n * n {
            self.out = vec![0.0; n * n];
        }
        let level = level.clamp(0.0, 1.0);
        let g = self.gain;

        match self.swing {
            SwingMode::Bipolar => {
                let s = level * 2.0 - 1.0;
                for i in 0..n * n {
                    self.out[i] = self.weights[i] * g * s;
                }
            }

            SwingMode::PingPong => {
                // Rate floors at a slow crawl so a silent passage parks the
                // oscillator instead of freezing it mid-swing.
                let hz = self.rate * (0.15 + 0.85 * level);
                self.phase = (self.phase + hz * dt).fract();
                // Triangle in -1..1, amplitude gated by the band.
                let tri = 1.0 - 4.0 * (self.phase - 0.5).abs();
                let s = tri * level;
                for i in 0..n * n {
                    self.out[i] = self.weights[i] * g * s;
                }
            }

            SwingMode::Springy => {
                if self.spring_x.len() != n * n {
                    self.spring_x = vec![0.0; n * n];
                    self.spring_v = vec![0.0; n * n];
                }
                // Substep so a frame hitch can't blow the integrator up.
                let steps = ((dt * 240.0).ceil() as usize).clamp(1, 8);
                let h = dt / steps as f32;
                let k = self.stiffness.max(1.0);
                let c = 2.0 * self.damping.max(0.0) * k.sqrt();
                for i in 0..n * n {
                    let target = self.weights[i] * g * level;
                    for _ in 0..steps {
                        let a = (target - self.spring_x[i]) * k - self.spring_v[i] * c;
                        self.spring_v[i] += a * h;
                        self.spring_x[i] += self.spring_v[i] * h;
                    }
                    self.out[i] = self.spring_x[i].clamp(-2.0, 2.0);
                }
            }
        }
    }

    /// Zero the integrator so re-arming doesn't inherit a mid-ring state.
    pub fn reset_state(&mut self) {
        self.phase = 0.0;
        for x in &mut self.spring_x {
            *x = 0.0;
        }
        for v in &mut self.spring_v {
            *v = 0.0;
        }
    }
}

/// Everything the layers produced this frame, ready to be written to the sim.
pub struct LayerOutputs {
    /// Force target, Add mode. Pre-depth.
    pub add: Vec<f32>,
    /// Force target, Scale mode. Pre-depth.
    pub scale: Vec<f32>,
    /// Reaction target, per cell.
    pub gate: Vec<GateState>,
    /// Peak drive across reaction layers with `drive_rate` set. `None` when no
    /// layer is asking to move the global probability, which is the signal to
    /// leave `reaction_probability` alone entirely.
    pub rate: Option<f32>,
}

impl LayerOutputs {
    pub fn new() -> Self {
        Self {
            add: Vec::new(),
            scale: Vec::new(),
            gate: Vec::new(),
            rate: None,
        }
    }

    /// Any reaction layer weighting any cell this frame?
    pub fn gates_active(&self) -> bool {
        self.gate.iter().any(|g| *g != GateState::Untouched)
    }

    fn reset(&mut self, n: usize) {
        self.add.clear();
        self.add.resize(n * n, 0.0);
        self.scale.clear();
        self.scale.resize(n * n, 0.0);
        self.gate.clear();
        self.gate.resize(n * n, GateState::Untouched);
        self.rate = None;
    }
}

impl Default for LayerOutputs {
    fn default() -> Self {
        Self::new()
    }
}

/// Fold every enabled layer into the output buffers.
///
/// Kept as a free function so `AudioMod::apply` can borrow `layers` mutably
/// while reading `bands` immutably without upsetting the borrow checker.
///
/// Gates resolve as OR: any layer holding a cell open wins. A cell weighted by
/// some layer but held open by none reads `Closed`; a cell no reaction layer
/// weights at all reads `Untouched` and keeps whatever was authored.
pub fn accumulate(
    layers: &mut [BandLayer],
    bands: &[f32],
    n: usize,
    dt: f32,
    out: &mut LayerOutputs,
) {
    out.reset(n);
    let cells = n * n;

    for layer in layers.iter_mut() {
        if !layer.enabled {
            continue;
        }
        layer.resize(n);
        let level = bands.get(layer.band).copied().unwrap_or(0.0);

        let target = layer.target;
        let combine = layer.combine;
        let threshold = layer.threshold;
        let drive_rate = layer.drive_rate;

        layer.drive(level, dt);
        // Both immutable now, so weights and out can be read side by side.
        let drive: &[f32] = &layer.out;
        let weights: &[f32] = &layer.weights;
        let span = cells.min(drive.len()).min(weights.len());

        match target {
            LayerTarget::Force => {
                let dst = match combine {
                    CombineMode::Add => &mut out.add,
                    CombineMode::Scale => &mut out.scale,
                };
                for i in 0..span.min(dst.len()) {
                    dst[i] += drive[i];
                }
            }

            LayerTarget::Reaction => {
                let mut peak = 0.0f32;
                for i in 0..span.min(out.gate.len()) {
                    if weights[i] == 0.0 {
                        continue;
                    }
                    // A negative weight produces negative drive when loud and
                    // positive drive when quiet, so this one comparison covers
                    // both "open on loud" and "open on quiet" without a branch.
                    if drive[i] >= threshold {
                        out.gate[i] = GateState::Open;
                        peak = peak.max(drive[i]);
                    } else if out.gate[i] == GateState::Untouched {
                        out.gate[i] = GateState::Closed;
                    }
                }
                if drive_rate {
                    out.rate = Some(out.rate.unwrap_or(0.0).max(peak));
                }
            }
        }
    }
}
