[# Particle Life Hyperdimensional 0.3.0

I made particle life and had ai help

## Run on Nobara / Fedora

With Rust/Cargo installed, install native build dependencies if needed:

```bash
sudo dnf install gcc alsa-lib-devel libxkbcommon-devel wayland-devel
cargo run --release --locked
```

Run from this directory, or use `./run.sh`. This is a source package, not an
installer. Existing creature data is read from `particle_life/book.json` in the
platform user-data directory.

## Multidimensional controls

Open **Dimensions & projection** at the top of Controls.

![New dimension controls and session inspector](docs/multidimensional-ui.png)

- **Physics dimensions:** choose 3–8. New axes receive seeded random coordinates;
  disabled axes are zeroed. Selection and trails reset.
- **Horizontal / Vertical / Depth:** choose three distinct displayed axes.
  Choosing an already displayed axis swaps it into that slot. XYZ and XYW are
  quick presets.
- **Rotate a plane:** select two axes and an angle. Rotations in different planes
  accumulate in a fixed order. Animation rotates the selected plane at a signed
  speed, even while physics is paused.
- **Hidden-axis slices:** each axis not displayed gets an enable toggle, center
  and width. Coordinates are measured after rotation. Show all removes clipping;
  Center slices restores domain-centered, domain-width slices.
- Particles, trails, rectangle/brush/depth selection share the same projection.
  Dragging maps the displayed plane back into the full-dimensional world.
  Projection changes do not alter particle coordinates or the force law.
- Wrapped-boundary trail segments are omitted instead of drawing a long line
  through the world.

## Playback

The default remains one physics step per displayed frame; `dt` retains its
original meaning and default.

- **Fixed steps / second:** wall-clock-paced simulation with a bounded catch-up
  budget. A warning appears if the requested rate cannot be sustained.
- **Display stride:** batch 1–16 steps before showing a state.
- **Strobe ×2:** compatibility shortcut overriding display stride to two.
- **Step / N:** while paused, always advance one step on either backend.

Audio analysis/modulation still runs once per UI frame. Fixed pacing improves
its speed relationship with physics; this is not sample-accurate or
bit-identical audio replay across display rates.

## Sessions, undo and inspection

Open **Session & creature inspector**.

- **Save / Load session:** validated JSON with particles and velocities,
  physics/backend, all three matrices, step and random sequence, projection,
  slices/camera, playback/trail settings and audio settings/file reference/
  playhead. Load restores paused. Audio is not embedded or automatically played;
  trail history restarts.
- **Undo edit / Redo edit:** eight in-memory simulation edit checkpoints. Restore
  pauses playback. This is not a continuous rewind recorder and does not undo
  changes to the Creature Book itself.
- **Random seed / Reset random sequence:** reproduce spawn and matrix
  randomization sequences within this pinned build. Sessions preserve the
  sequence counter. Cross-device bit identity is not promised.
- **Inspect selection:** sample selected particles or the whole world. Shows
  populations, center speed, RMS spread, per-axis centers, and population inside
  hidden-axis slices. Values show the sampled step; inspection does not force
  continuous GPU readback.
- **A/B checkpoints:** capture two worlds, restore either paused, and compare
  translation-aligned position/velocity RMS residuals. Compare the same particles
  in the same array order. A small residual suggests a return after the step
  difference, not proof of a shortest period; also check particle types. Slots
  are in-memory. Save a session to preserve an experiment across restarts.

Saves use a synced temporary file and rename, retaining a `.bak` of the previous
file. Failed Creature Book saves retain dirty state and retry after three seconds
with visible errors. Session data is validated before state replacement.

## Creature Book

Old 3D and higher-dimensional books remain readable. New live captures preserve
velocities, physics settings, backend, dimension, reaction rules and force matrix.
Disable **Include velocities** to save a shape instead of a live particle state.

**Restore recorded physics when spawning** is off by default. Enabling it changes
world-wide physics/rules/backend to the recorded environment and disarms audio
modulation. Otherwise spawning uses the current world rules. Compatible
surroundings are still necessary to reproduce a creature's behavior.

Compact creatures crossing a wrapped boundary are unwrapped before computing
relative coordinates. Structures spanning more than half a wrapped axis are
ambiguous under reference-point unwrapping.

## Correctness and compatibility

- Current GPU state is read before an edit transaction. No post-edit readback
  overwrites movement/velocity edits or remaps deleted-particle survivors.
- Backend transitions synchronize current state and reset selection/trails.
- Reactions run after integration, reading every particle at the same time.
- CPU/GPU share a deterministic directed-pair probability hash. Probability zero
  never passes; probability one always passes.
- Competing reactions choose the greatest passing neighbor index, independent
  of GPU grid insertion order.
- Each GPU trace timer has one writer. Incoming and outgoing reactions can
  trigger both particles without cross-thread write races. CPU-triggered trails
  are supported too.
- Every GPU substep receives its own frame seed in queue submission order.
- Type-count reduction removes reaction outputs outside the new type range.

**The historical CPU/GPU unit difference is retained:** CPU radius, mixing radius
and max speed scale with `bounds / 20`; GPU uses raw values. The UI reports the
effective radius. Switching backends can change dynamics unless units match.

Correcting reactions changes old reaction-enabled behavior. Undefined timer races
and mixed-time reaction distances are not emulated. Original force-law, friction,
step-size and unit defaults remain. Old creature entries cannot retroactively
recover velocities or settings that were never stored.

## Performance

The default GPU force path retains the original ascending-index all-pairs scan.
**GPU neighbor grid** is opt-in because changing force accumulation order can
change sensitive floating-point dynamics.

The grid builds linked cell lists using XYZ as a conservative broad phase,
then checks the full 3D–8D distance. It rebuilds after integration for reactions.
Omitting extra axes from the broad phase admits extra candidates, never excludes
full-distance neighbors. Resolution is capped at 64 cells per axis. Fewer than
three cells per axis falls back to all-pairs. Large radii and dense clusters can
still be expensive; a grid is not a frame-time guarantee.

GPU reactions remain suspended above 10,000 particles when no usable local grid
is active. The UI reports this condition. All spawn paths retain the 50,000
particle cap. Renderer/trail capacity remains 200,000 with device-derived
trail-length limits.

Interactive CPU edits and saves still need GPU readback. Simply leaving a
selection active no longer forces a full readback every frame.

## Controls and validation

- `Space`: pause/resume; `N`: single step
- `M`: matrix editor; `B`: Creature Book; `T`: cycle trails
- `1`: rectangle; `2`: brush; `3`: displayed-depth slice
- Camera mode: WASD, Q/E, right-mouse look
- Selection dragging: middle mouse in the displayed/rotated plane

```bash
cargo test --locked
# Requires a graphics or software Vulkan/Metal/DX12 adapter:
cargo test --locked -- --include-ignored
cargo build --release --locked
```

See [VALIDATION.md](VALIDATION.md) for the verification record and remaining
checks on your Nobara/NVIDIA desktop.
](https://github.com/emilstekcor/particle-life)
