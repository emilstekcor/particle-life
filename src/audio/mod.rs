//! Audio-driven modulation of the force matrix and reaction table.
//!
//! Flow: file -> symphonia decode to mono f32 in memory -> cpal output stream
//! reads it and publishes a playhead -> each UI frame we FFT a window at the
//! playhead, fold bins into log-spaced bands, envelope-follow + normalize them,
//! then let each band layer turn its level into per-cell drive values which are
//! written on top of a snapshot of the authored state.
//!
//! The authored matrix is snapshotted when modulation is armed and restored when
//! disarmed, so audio never eats the user's edits.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

mod layers;
pub use layers::{
    accumulate, BandLayer, CombineMode, GateState, LayerOutputs, LayerTarget, SwingMode,
};

pub const BANDS: usize = 8;
const FFT_SIZE: usize = 2048;

/// Log-spaced band edges in Hz (BANDS + 1 entries).
const BAND_EDGES: [f32; BANDS + 1] = [
    30.0, 80.0, 160.0, 320.0, 640.0, 1300.0, 2600.0, 6000.0, 16000.0,
];

pub const BAND_NAMES: [&str; BANDS] = [
    "sub", "bass", "low-mid", "mid", "hi-mid", "presence", "brilliance", "air",
];

// ── Superseded by BandLayer (src/audio/layers.rs) ───────────────────────────
// A `Route` was one band driving one matrix cell — i.e. a single non-zero cell
// of a band's weight matrix. `BandLayer` holds the whole matrix, so the old
// model round-trips into the new one without loss. Kept for reference against
// any older saved profiles.
//
// #[derive(Clone, Copy, Debug)]
// pub struct Route {
//     pub row: usize,
//     pub col: usize,
//     pub band: usize,
//     /// Multiplier applied on top of the global depth. Negative inverts.
//     pub gain: f32,
//     /// If true the band drives -1..1 instead of 0..1.
//     pub bipolar: bool,
//     pub enabled: bool,
// }
//
// impl Route {
//     pub fn new(row: usize, col: usize, band: usize) -> Self {
//         Self { row, col, band, gain: 1.0, bipolar: false, enabled: true }
//     }
// }

pub struct AudioMod {
    // ── Source ──────────────────────────────────────────────────────────────
    pub file_name: String,
    pub path: Option<PathBuf>,
    pub load_error: Option<String>,
    samples: Arc<Vec<f32>>,
    sample_rate: u32,

    // ── Transport ───────────────────────────────────────────────────────────
    playhead: Arc<AtomicUsize>,
    stream: Option<cpal::Stream>,
    /// Output gain as f32 bits, read by the audio callback. Muting only affects
    /// what you hear — analysis always runs on the decoded samples.
    gain: Arc<AtomicU32>,
    pub playing: bool,
    pub muted: bool,
    pub volume: f32,
    pub looping: bool,

    // ── Analysis ────────────────────────────────────────────────────────────
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    /// Post-envelope, post-normalization band levels in 0..1. Read this for UI.
    pub bands: [f32; BANDS],
    band_env: [f32; BANDS],
    band_peak: [f32; BANDS],
    pub attack: f32,
    pub release: f32,

    // ── Modulation ──────────────────────────────────────────────────────────
    pub armed: bool,
    pub depth: f32,
    /// 0 = instant (jittery), 1 = frozen. ~0.85 is a good default.
    pub smoothing: f32,
    /// One layer per band. Each owns an N×N weight matrix.
    pub layers: Vec<BandLayer>,
    /// What the layers produced this frame. The UI reads gate state from here.
    pub outputs: LayerOutputs,
    /// (layer, row, col) hovered last frame, for cross-highlighting.
    pub hovered: Option<(usize, usize, usize)>,
    /// Show the read-only base snapshot in the audio tab.
    pub show_base: bool,

    /// Snapshot of the authored force matrix taken at arm time.
    base: Vec<f32>,
    base_types: usize,
    /// Snapshot of the authored reaction table taken at arm time.
    base_rx: Vec<i32>,
    /// Snapshot of the authored global reaction probability.
    base_prob: f32,
    /// Last force values we actually wrote, for slew limiting.
    applied: Vec<f32>,
}

impl AudioMod {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let window = (0..FFT_SIZE)
            .map(|i| {
                // Hann
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos())
            })
            .collect();

        Self {
            file_name: String::new(),
            path: None,
            load_error: None,
            samples: Arc::new(Vec::new()),
            sample_rate: 44100,

            playhead: Arc::new(AtomicUsize::new(0)),
            stream: None,
            gain: Arc::new(AtomicU32::new(0.8f32.to_bits())),
            playing: false,
            muted: false,
            volume: 0.8,
            looping: true,

            fft,
            window,
            scratch: vec![Complex { re: 0.0, im: 0.0 }; FFT_SIZE],
            bands: [0.0; BANDS],
            band_env: [0.0; BANDS],
            band_peak: [1e-4; BANDS],
            attack: 0.002,
            release: 0.18,

            armed: false,
            depth: 0.15,
            smoothing: 0.85,
            layers: Vec::new(),
            outputs: LayerOutputs::new(),
            hovered: None,
            show_base: false,

            base: Vec::new(),
            base_types: 0,
            base_rx: Vec::new(),
            base_prob: 0.1,
            applied: Vec::new(),
        }
    }

    pub fn has_file(&self) -> bool {
        !self.samples.is_empty()
    }

    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate.max(1) as f32
    }

    pub fn position_secs(&self) -> f32 {
        self.playhead.load(Ordering::Relaxed) as f32 / self.sample_rate.max(1) as f32
    }

    pub fn seek_secs(&mut self, t: f32) {
        let idx = (t.max(0.0) * self.sample_rate as f32) as usize;
        self.playhead
            .store(idx.min(self.samples.len().saturating_sub(1)), Ordering::Relaxed);
    }

    // ── Loading ─────────────────────────────────────────────────────────────

    /// Open a native file dialog and load the chosen file. Returns true on success.
    pub fn pick_file(&mut self) -> bool {
        let picked = rfd::FileDialog::new()
            .add_filter(
                "Audio / video",
                &["mp3", "wav", "flac", "ogg", "m4a", "mp4", "aac", "opus"],
            )
            .add_filter("All files", &["*"])
            .set_title("Select an audio or video file")
            .pick_file();

        match picked {
            Some(p) => self.load_path(&p),
            None => false,
        }
    }

    pub fn load_path(&mut self, path: &Path) -> bool {
        self.stop();
        match decode_to_mono(path) {
            Ok((samples, rate)) => {
                if samples.is_empty() {
                    self.load_error = Some("File decoded to zero samples".into());
                    return false;
                }
                self.samples = Arc::new(samples);
                self.sample_rate = rate;
                self.playhead.store(0, Ordering::Relaxed);
                self.file_name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                self.path = Some(path.to_path_buf());
                self.load_error = None;
                self.band_peak = [1e-4; BANDS];
                true
            }
            Err(e) => {
                self.load_error = Some(e);
                false
            }
        }
    }

    // ── Transport ───────────────────────────────────────────────────────────

    pub fn play(&mut self) {
        if !self.has_file() || self.playing {
            return;
        }
        self.push_gain();
        match build_stream(
            self.samples.clone(),
            self.sample_rate,
            self.playhead.clone(),
            self.gain.clone(),
            self.looping,
        ) {
            Ok(s) => {
                self.stream = Some(s);
                self.playing = true;
            }
            Err(e) => {
                // No output device is not fatal — we can still analyze silently by
                // advancing the playhead ourselves in update().
                self.load_error =
                    Some(format!("audio output unavailable: {e} (analyzing silently)"));
                self.playing = true;
            }
        }
    }

    pub fn pause(&mut self) {
        self.stream = None;
        self.playing = false;
    }

    pub fn stop(&mut self) {
        self.stream = None;
        self.playing = false;
        self.playhead.store(0, Ordering::Relaxed);
    }

    pub fn toggle_play(&mut self) {
        if self.playing {
            self.pause();
        } else {
            self.play();
        }
    }

    // ── Analysis ────────────────────────────────────────────────────────────

    /// Run one analysis frame. `dt` is the render frame time in seconds.
    pub fn analyze(&mut self, dt: f32) {
        if !self.has_file() {
            for b in 0..BANDS {
                self.band_env[b] *= 0.9;
                self.bands[b] = self.band_env[b];
            }
            return;
        }

        // If there's no output stream (device failed) advance the playhead so the
        // analysis still moves through the file.
        if self.playing && self.stream.is_none() {
            let adv = (dt * self.sample_rate as f32) as usize;
            let next = self.playhead.load(Ordering::Relaxed) + adv;
            let next = if next >= self.samples.len() {
                if self.looping {
                    next % self.samples.len().max(1)
                } else {
                    self.playing = false;
                    self.samples.len() - 1
                }
            } else {
                next
            };
            self.playhead.store(next, Ordering::Relaxed);
        }

        let head = self.playhead.load(Ordering::Relaxed);

        // Center the analysis window on the playhead.
        let start = head.saturating_sub(FFT_SIZE / 2);
        for i in 0..FFT_SIZE {
            let s = self.samples.get(start + i).copied().unwrap_or(0.0);
            self.scratch[i] = Complex {
                re: s * self.window[i],
                im: 0.0,
            };
        }
        self.fft.process(&mut self.scratch);

        // Fold bins into log-spaced bands. Only the first half is meaningful.
        let bin_hz = self.sample_rate as f32 / FFT_SIZE as f32;
        let mut raw = [0.0f32; BANDS];
        let mut counts = [0u32; BANDS];
        for bin in 1..FFT_SIZE / 2 {
            let hz = bin as f32 * bin_hz;
            if hz < BAND_EDGES[0] || hz >= BAND_EDGES[BANDS] {
                continue;
            }
            let mut b = 0;
            while b + 1 < BANDS && hz >= BAND_EDGES[b + 1] {
                b += 1;
            }
            raw[b] += self.scratch[bin].norm();
            counts[b] += 1;
        }

        // Envelope follow, then normalize against a slowly decaying per-band peak
        // so quiet passages still produce usable motion.
        let atk = 1.0 - (-dt / self.attack.max(1e-4)).exp();
        let rel = 1.0 - (-dt / self.release.max(1e-4)).exp();

        for b in 0..BANDS {
            let v = if counts[b] > 0 {
                (raw[b] / counts[b] as f32).sqrt()
            } else {
                0.0
            };
            let coef = if v > self.band_env[b] { atk } else { rel };
            self.band_env[b] += (v - self.band_env[b]) * coef;

            self.band_peak[b] = (self.band_peak[b] * 0.9995).max(self.band_env[b]).max(1e-4);
            self.bands[b] = (self.band_env[b] / self.band_peak[b]).clamp(0.0, 1.0);
        }
    }

    // ── Modulation ──────────────────────────────────────────────────────────

    /// Snapshot the current authored state as the base to modulate around.
    pub fn capture_base(&mut self, sim: &crate::sim::SimState) {
        self.base = sim.force_matrix.clone();
        self.base_types = sim.params.type_count;
        self.applied = self.base.clone();
        self.base_rx = sim.reaction_table.clone();
        self.base_prob = sim.params.reaction_probability;
    }

    /// Write the snapshot back, undoing all modulation.
    pub fn restore_base(&self, sim: &mut crate::sim::SimState) {
        if self.base_types != sim.params.type_count {
            return;
        }
        for i in 0..self.base_types {
            for j in 0..self.base_types {
                sim.set_rule(i, j, self.base[i * self.base_types + j]);
            }
        }
        if self.base_rx.len() == self.base_types * self.base_types {
            for i in 0..self.base_types {
                for j in 0..self.base_types {
                    sim.set_reaction(i, j, self.base_rx[i * self.base_types + j]);
                }
            }
            sim.set_reaction_probability(self.base_prob);
        }
    }

    /// Read-only view of the force snapshot being modulated around.
    pub fn base_snapshot(&self) -> &[f32] {
        &self.base
    }

    /// The reaction a pair was authored with, ignoring any live gating.
    pub fn base_reaction(&self, row: usize, col: usize) -> i32 {
        let n = self.base_types;
        if row >= n || col >= n {
            return -1;
        }
        self.base_rx.get(row * n + col).copied().unwrap_or(-1)
    }

    /// Does any layer target the reaction table?
    pub fn has_reaction_layers(&self) -> bool {
        self.layers
            .iter()
            .any(|l| l.target == LayerTarget::Reaction)
    }

    pub fn arm(&mut self, sim: &crate::sim::SimState) {
        self.capture_base(sim);
        for l in &mut self.layers {
            l.resize(sim.params.type_count);
            l.reset_state();
        }
        self.armed = true;
    }

    pub fn disarm(&mut self, sim: &mut crate::sim::SimState) {
        if self.armed {
            self.restore_base(sim);
        }
        self.armed = false;
    }

    /// Fold every layer onto the snapshot and write the result into the sim.
    ///
    /// Force layers are continuous: Add layers offset the base, Scale layers
    /// multiply it, and both accumulate independently so a cell driven by one
    /// of each gets `base * (1 + scale*depth) + add*depth`. The two toggles
    /// compose instead of fighting.
    ///
    /// Reaction layers are discrete. A reaction cell is an i32 naming the type
    /// a pair becomes, so there's nothing to offset — instead the gate decides
    /// whether the authored reaction is live or reads -1 this frame. Cells no
    /// reaction layer weights are written straight back from the snapshot, so
    /// arming doesn't quietly freeze the table.
    fn apply(&mut self, sim: &mut crate::sim::SimState, dt: f32) {
        let n = sim.params.type_count;
        if !self.armed || self.base_types != n || self.base.len() != n * n {
            return;
        }
        if self.applied.len() != n * n {
            self.applied = self.base.clone();
        }

        let mut outputs = std::mem::take(&mut self.outputs);
        accumulate(&mut self.layers, &self.bands, n, dt, &mut outputs);

        // ── Force matrix ────────────────────────────────────────────────
        let keep = self.smoothing.clamp(0.0, 0.99);
        let depth = self.depth;

        for i in 0..n * n {
            let scaled = self.base[i] * (1.0 + outputs.scale[i] * depth);
            let target = (scaled + outputs.add[i] * depth).clamp(-1.0, 1.0);
            self.applied[i] += (target - self.applied[i]) * (1.0 - keep);
        }

        for i in 0..n {
            for j in 0..n {
                sim.set_rule(i, j, self.applied[i * n + j]);
            }
        }

        // ── Reaction table ──────────────────────────────────────────────
        // Only touch it if a layer actually targets reactions; otherwise
        // arming the audio would lock the Reactions tab out of editing.
        // No slew here — a gate is on or off, and easing it would just
        // produce a frame or two of a wrong reaction product.
        if self.has_reaction_layers() && self.base_rx.len() == n * n {
            for i in 0..n {
                for j in 0..n {
                    let idx = i * n + j;
                    let want = match outputs.gate[idx] {
                        GateState::Closed => -1,
                        // Untouched cells fall through to whatever was authored.
                        GateState::Open | GateState::Untouched => self.base_rx[idx],
                    };
                    sim.set_reaction(i, j, want);
                }
            }

            if let Some(rate) = outputs.rate {
                sim.set_reaction_probability((self.base_prob + rate * depth).clamp(0.0, 1.0));
            }
        }

        self.outputs = outputs;
    }

    /// Publish the current mute/volume state to the audio callback.
    pub fn push_gain(&self) {
        let g = if self.muted { 0.0 } else { self.volume.clamp(0.0, 1.0) };
        self.gain.store(g.to_bits(), Ordering::Relaxed);
    }

    /// Call once per frame. Handles analysis and matrix writing.
    pub fn tick(&mut self, sim: &mut crate::sim::SimState, dt: f32) {
        self.push_gain();
        self.analyze(dt);
        if self.armed {
            self.apply(sim, dt);
        }
    }

    // ── Layer helpers ───────────────────────────────────────────────────────

    /// Seed one channel per band with an alternating-sign off-diagonal
    /// pattern. A starting point to edit, not a preset.
    pub fn auto_fill(&mut self, type_count: usize) {
        self.layers.clear();
        for b in 0..BANDS {
            let mut l = BandLayer::new(b, type_count);
            for i in 0..type_count {
                for j in 0..type_count {
                    if i == j {
                        continue; // self-interaction is usually load-bearing
                    }
                    // Each band takes a diagonal stripe of the matrix so the
                    // channels don't all drive the same cells.
                    if (i + j + b) % BANDS == 0 {
                        let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                        l.set(i, j, sign);
                    }
                }
            }
            self.layers.push(l);
        }
    }
}

impl Default for AudioMod {
    fn default() -> Self {
        Self::new()
    }
}

// ── Decoding ────────────────────────────────────────────────────────────────

/// Decode any symphonia-supported container/codec to interleaved-averaged mono.
fn decode_to_mono(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open failed: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("unsupported or corrupt file: {e}"))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| "no decodable audio track in this file".to_string())?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("no decoder for this codec: {e}"))?;

    let mut rate = track.codec_params.sample_rate.unwrap_or(44_100);
    let mut out: Vec<f32> = Vec::new();
    let mut sbuf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break, // end of stream (or unrecoverable) — keep what we have
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                rate = spec.rate;
                let dur = audio_buf.capacity() as u64;

                let need_new = match &sbuf {
                    Some(sb) => (sb.capacity() as u64) < dur * spec.channels.count() as u64,
                    None => true,
                };
                if need_new {
                    sbuf = Some(SampleBuffer::<f32>::new(dur, spec));
                }
                let sb = sbuf.as_mut().unwrap();
                sb.copy_interleaved_ref(audio_buf);

                let ch = spec.channels.count().max(1);
                for frame in sb.samples().chunks(ch) {
                    out.push(frame.iter().sum::<f32>() / ch as f32);
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    if out.is_empty() {
        return Err("decoded no audio samples".into());
    }
    Ok((out, rate))
}

// ── Playback ────────────────────────────────────────────────────────────────

fn build_stream(
    samples: Arc<Vec<f32>>,
    src_rate: u32,
    playhead: Arc<AtomicUsize>,
    gain: Arc<AtomicU32>,
    looping: bool,
) -> Result<cpal::Stream, String> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default output device".to_string())?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("no default output config: {e}"))?;

    let out_rate = supported.sample_rate().0 as f64;
    let channels = supported.channels() as usize;
    let step = src_rate as f64 / out_rate;
    let config: cpal::StreamConfig = supported.config();

    // Fractional read position, kept in the callback; the atomic playhead is the
    // integer view of it that the analysis thread reads.
    let mut pos = playhead.load(Ordering::Relaxed) as f64;
    let len = samples.len();

    let err_fn = |e| log::error!("audio stream error: {e}");

    macro_rules! make {
        ($t:ty, $conv:expr) => {{
            let samples = samples.clone();
            let playhead = playhead.clone();
            let gain = gain.clone();
            device
                .build_output_stream(
                    &config,
                    move |data: &mut [$t], _: &cpal::OutputCallbackInfo| {
                        let g = f32::from_bits(gain.load(Ordering::Relaxed));
                        for frame in data.chunks_mut(channels) {
                            let i = pos as usize;
                            let s = if i + 1 < len {
                                // linear interpolation between neighbours
                                let f = (pos - i as f64) as f32;
                                samples[i] * (1.0 - f) + samples[i + 1] * f
                            } else {
                                0.0
                            };
                            let v = $conv(s * g);
                            for ch in frame.iter_mut() {
                                *ch = v;
                            }
                            pos += step;
                            if pos as usize >= len {
                                if looping {
                                    pos = 0.0;
                                } else {
                                    pos = (len.saturating_sub(1)) as f64;
                                }
                            }
                        }
                        playhead.store((pos as usize).min(len.saturating_sub(1)), Ordering::Relaxed);
                    },
                    err_fn,
                    None,
                )
                .map_err(|e| format!("build_output_stream: {e}"))?
        }};
    }

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => make!(f32, |s: f32| s),
        cpal::SampleFormat::I16 => make!(i16, |s: f32| (s.clamp(-1.0, 1.0) * 32767.0) as i16),
        cpal::SampleFormat::U16 => {
            make!(u16, |s: f32| ((s.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0) as u16)
        }
        f => return Err(format!("unsupported sample format {f:?}")),
    };

    stream.play().map_err(|e| format!("stream.play: {e}"))?;
    Ok(stream)
}
