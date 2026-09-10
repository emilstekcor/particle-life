# Particle Life Hyperdimensional

General-use guide for version **0.3.0** — Windows, macOS, and Linux.

A desktop particle-life sandbox with 3–8 dimensional physics, editable interaction rules, trails, audio modulation, and a Creature Book for saving interesting structures.

This is a source release. You build and launch it with Cargo, Rust's build tool. The same launch command works on all three operating systems:

```sh
cargo run --release --locked
```

## Contents

- [Installation](#installation)
- [Running the app](#running-the-app)
- [First five minutes](#first-five-minutes)
- [Controls](#controls)
- [Working in multiple dimensions](#working-in-multiple-dimensions)
- [Rules and audio](#rules-and-audio)
- [Saving your work](#saving-your-work)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Build and validation notes](#build-and-validation-notes)

## Installation

You need Rust/Cargo, your operating system's native build tools, and a working graphics driver. Internet access is needed for the initial Rust installation and dependency downloads. No Python, Node.js, or separate game engine is required.

The app uses wgpu for graphics and compute. Even when you select CPU physics, rendering and other GPU operations still require a compatible graphics adapter.

These are setup instructions for the project's cross-platform dependencies. Version 0.3.0 has been built and tested in a Linux environment using software Vulkan; native Windows and macOS builds have not yet been validated.

### Windows

1. Download the appropriate Windows installer from the [official Rust installation page](https://rust-lang.org/tools/install/).
2. Run it and follow the prompts for the default stable MSVC toolchain and its prerequisites.
3. If installing Visual Studio components manually, select **Desktop development with C++**, including the MSVC compiler tools and Windows SDK. See the [official Rust Windows prerequisites](https://rust-lang.github.io/rustup/installation/windows-msvc.html).
4. Close and reopen PowerShell after installation.
5. Check that both commands work:

```powershell
rustc --version
cargo --version
```

Extract the entire application ZIP with **Extract All**. Open PowerShell in the extracted directory containing `Cargo.toml`. There may be an extra outer folder created by the ZIP extractor.

```powershell
cd "C:\path\to\particle-life-hyperdimensional"
cargo run --release --locked
```

Replace the example path with your actual folder. Native Windows PowerShell is sufficient; WSL and `run.sh` are not required.

### macOS

Install Apple's Command Line Tools from Terminal:

```sh
xcode-select --install
```

Finish the installer before continuing. If the tools are already installed, macOS will tell you. See [Apple's command-line tools instructions](https://developer.apple.com/library/archive/technotes/tn2339/_index.html).

Install Rust using the command published on the [official Rust installation page](https://rust-lang.org/tools/install/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Choose the default stable installation, then open a new Terminal window and check:

```sh
rustc --version
cargo --version
```

Extract the entire application ZIP, then enter the folder containing `Cargo.toml`:

```sh
cd "/path/to/particle-life-hyperdimensional"
cargo run --release --locked
```

On Apple silicon, use a normal native Terminal session and native Rust installation. Intel Macs need an Intel toolchain. Cargo builds for the host by default. This source package does not include a signed `.app` or a universal Mac binary.

A mouse with a middle button is useful: this version uses middle-button dragging to move selected particles and does not provide a dedicated trackpad replacement gesture.

### Linux

Install the native build dependencies for your distribution first. The following are starting package sets for common desktop installations; package names may vary by distribution release.

**Fedora / Nobara:**

```sh
sudo dnf install gcc pkgconf-pkg-config alsa-lib-devel libxkbcommon-devel wayland-devel
```

**Ubuntu / Debian:**

```sh
sudo apt update
sudo apt install build-essential pkg-config libasound2-dev libxkbcommon-dev libwayland-dev
```

**Arch Linux:**

```sh
sudo pacman -S --needed base-devel pkgconf alsa-lib libxkbcommon wayland
```

Install Rust using the [official installer](https://rust-lang.org/tools/install/):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Open a new terminal, verify the installation, then launch from the extracted folder containing `Cargo.toml`:

```sh
rustc --version
cargo --version
cd "/path/to/particle-life-hyperdimensional"
cargo run --release --locked
```

Use your distribution's graphics driver packages. File pickers also rely on the desktop's portal services when using the default Linux dialog backend; see troubleshooting if dialogs do not open.

## Running the app

Always run Cargo from the folder containing `Cargo.toml`:

```sh
cargo run --release --locked
```

- `--release` enables compiler optimizations, which matter for simulation performance.
- `--locked` keeps the dependency versions recorded in the included `Cargo.lock`.
- The first build takes longer while Cargo downloads and compiles dependencies. Later launches reuse the build unless something changes.
- Keep the terminal open while the app runs so you can see errors.

The included `run.sh` is an optional convenience wrapper for Unix-like systems. Launching directly with Cargo is fully supported.

To compile without launching:

```sh
cargo build --release --locked
```

The resulting executable is `target/release/particle-life-hyperdimensional.exe` on Windows or `target/release/particle-life-hyperdimensional` on macOS/Linux. An executable built on one OS is not a cross-platform executable.

## First five minutes

1. Launch the app and press **Space** to pause while exploring.
2. Find the **Controls** window. Expand sections and scroll inside the panel to reach the rest of the options. Drag window title bars to arrange panels on smaller screens.
3. In the spawn controls, use a modest population to start. **Even split** distributes the available allocation among particle types; **Respawn** rebuilds the world from the chosen mix.
4. Open **Matrix Editor** with **M** and try a force preset or random rules.
5. Press **Space** to resume. Change one setting at a time to see its effect.
6. Press **T** to cycle trail modes and make motion easier to follow.
7. Pause when you find something interesting. Open **Session & creature inspector** and use **Save session…** before experimenting further.

Respawning replaces the current population. Save a session first if you want to return to it.

## Controls

Use mouse tools over the simulation view, outside UI panels. Most shortcuts are ignored while typing in a text field.

| Input | Action |
| --- | --- |
| Space | Pause / resume physics |
| N | Advance one physics step while paused |
| M | Toggle Matrix Editor |
| B | Toggle Creature Book |
| T | Cycle trail rendering modes |
| 1 | Rectangle selection tool |
| 2 | Brush selection tool |
| 3 | Displayed-depth slice selection tool |
| Left mouse drag | Use the active selection tool with Camera Mode off |
| Middle mouse drag | Move selected particles with Camera Mode off |
| W / S | Move forward / backward with Camera Mode on |
| A / D | Move left / right with Camera Mode on |
| Q / E | Move down / up with Camera Mode on |
| Right mouse drag | Look around with Camera Mode on |
| Mouse wheel | Adjust fly speed with Camera Mode on |

Toggle **Camera Mode** in the **Camera** section. Turn it off to select or move particles. **Reset pos** returns the camera to a useful starting position.

The **Selection** section provides brush radius and depth-slice settings, plus actions such as clear, delete, duplicate, change type, and adjust velocity. Pause first for precise editing. **Freeze** zeros selected velocities; forces can move those particles again when simulation resumes.

### Playback speed

By default, one physics step runs per displayed frame. **Fixed steps / second** requests a wall-clock simulation rate, subject to available processing time. **Display stride** batches steps before displaying a state.

**Strobe ×2** overrides the stride with two steps. It can make a two-step oscillation appear stationary. Pause and press **N** to inspect the other phase.

Changing physics `dt` changes the integration itself. Use playback controls when you want to adjust observation speed without changing that parameter.

## Working in multiple dimensions

Open **Dimensions & projection** near the top of Controls.

**Physics dimensions** changes the simulated world, from 3D to 8D. The axis names are X, Y, Z, W, V, U, T, and S. Increasing the dimension adds seeded random coordinates on the new axes; reducing it zeros disabled axes. Changing dimension resets selection and trails, so save first when preserving a state matters.

**Horizontal**, **Vertical**, and **Depth** choose which three axes you view. Choosing an axis already in another slot swaps their positions. **XYZ** and **XYW** are quick presets.

**Rotate a plane** mixes two axes for viewing. Choose a plane, adjust its angle, or enable animation. Projection rotation changes what you see without changing the physical particle coordinates. Animation can keep rotating the view while physics is paused.

**Hidden-axis slices** filter particles by coordinates on axes outside the displayed three. Each has an enable toggle, center, and width. The filter is measured after projection rotation. Use **Show all** to remove hidden-axis clipping.

### A simple 4D experiment

1. Save the current session and pause.
2. Set **Physics dimensions** to **4**.
3. Choose **XYZ**, then **XYW**. You are viewing different coordinates of the same world.
4. Choose the **X–W** rotation plane and move its angle slowly.
5. Enable a hidden-axis slice and narrow its width to inspect part of the world.
6. Use **Show all** if too much disappears, then resume the physics.

Particles that appear close in a projection may be far apart along a hidden axis. Interactions use the full-dimensional distance. The **Slice** selection tool filters displayed depth; hidden-axis slices are separate visibility controls.

Particles, trails, and selection use the same projection. Moving a selection maps the visible drag back into the full-dimensional world.

## Rules and audio

Use **Matrix Editor** to adjust type-to-type interactions. Positive force values attract in the interaction region; negative values repel. Close-range separation also depends on the force law. Interactions can be asymmetric, so one type's response to another need not match the reverse response.

Reaction rules can change particle types when interaction conditions are met. Save a session before changing the number of types or replacing a rule set.

For audio, find the audio controls and choose **Load audio…**, then start playback. Enable **Drive force matrix** to arm modulation and configure the bands/channels and their targets. Loading or playing a file alone is distinct from enabling modulation. Start with small modulation amounts so the underlying behavior remains easy to observe.

Audio-driven behavior depends on UI-frame timing. Saved audio settings do not guarantee identical playback-driven simulations across computers.

## Saving your work

### Sessions: save the whole experiment

Open **Session & creature inspector**, then choose **Save session…** or **Load session…**. Pick your own location for the JSON file.

Sessions include particles and velocities, physics/backend, matrices, random sequence, projection and camera, playback/trail settings, and audio settings with a file reference and playhead.

A loaded session starts paused. Trail history restarts. Audio files are not embedded and do not autoplay. When moving a session to another computer, copy any audio files too and reload the audio if its old path no longer exists.

**Undo edit / Redo edit** keeps up to eight simulation edit checkpoints in memory. It is not a continuous rewind or an undo system for Creature Book entries.

**A/B checkpoints** let you capture, compare, and restore two states during an experiment. They disappear when the app closes. Use session files for lasting saves. A small comparison residual is evidence of similarity, not proof of an exact repeating period.

### Creature Book: save a selected structure

1. Pause and turn Camera Mode off.
2. Select the particles you want to keep.
3. Press **B**, enter a name, and click **Save**.
4. Keep **Include velocities (live creature)** enabled to retain motion, or disable it to save a shape.
5. Select a saved entry and use **Spawn selected creature** to insert it.

**Restore recorded physics when spawning** is off by default. Enabling it applies recorded settings to the entire world and disarms audio modulation. A creature also needs compatible surroundings to reproduce its behavior.

The Creature Book saves automatically to `book.json` in these normal locations:

| OS | Default location |
| --- | --- |
| Windows | `%APPDATA%\particle_life\book.json` |
| macOS | `~/Library/Application Support/particle_life/book.json` |
| Linux | `~/.local/share/particle_life/book.json` |

On Linux, a configured `XDG_DATA_HOME` replaces `~/.local/share`. These locations follow the project's [dirs data-directory rules](https://docs.rs/dirs/5.0.1/dirs/fn.data_dir.html).

To migrate a book, close the app on both computers, back up the destination book, then copy the source `book.json` into the destination's location. This replaces that book; it does not merge entries. Keep a separate backup of important sessions and books. Save operations retain a `.bak` of the previous file when replacing an existing save.

## Performance

Start with a small population and increase it gradually. Long trails, dense clusters, large interaction radii, and additional dimensions can increase cost. The 50,000-particle spawn cap is a limit, not a performance target.

**GPU neighbor grid** is optional. It can reduce neighbor-search work, but the benefit depends on the scene. Large radii can make it fall back to all-pairs processing. Changing force accumulation order can also change sensitive dynamics; keep it off when comparing against the default behavior.

Without a usable local grid, GPU reactions are suspended above 10,000 particles. Watch the UI's status messages.

CPU and GPU physics retain a historical unit difference: CPU interaction radius, mixing radius, and max speed scale with `bounds / 20`; GPU values are unscaled. Switching backends can therefore change behavior. Save before switching during a delicate experiment.

GPU edits and saves may briefly pause while particle data is read back. For slow rendering, first reduce population and trail length or turn trails off.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `cargo` is not found | Reopen the terminal after installing Rust. Check `cargo --version`; ensure Rust's Cargo bin directory is on PATH. |
| Cargo cannot find `Cargo.toml` | Change into the inner project folder after extracting the ZIP. |
| Windows reports missing `link.exe` or SDK libraries | Complete the MSVC C++ workload and Windows SDK installation, then reopen the terminal. |
| macOS reports missing compiler tools or SDK | Run `xcode-select --install` and finish the installer. |
| Linux reports missing `alsa` or `alsa.pc` | Install ALSA development files and pkg-config using the distro instructions above. Audio support is part of this build even if you do not play music. |
| App fails while creating the graphics adapter/device | Check your graphics driver and run in a local graphical desktop session. CPU physics does not bypass GPU initialization. |
| Linux file picker does not open | Check that `xdg-desktop-portal` and your desktop's matching portal backend are installed and running. |
| Nothing appears after changing dimensions | Choose **Show all**, try **XYZ**, reset the camera, and check the particle count. |
| Mouse selects when you want to look around | Turn Camera Mode on. Turn it off again to edit selections. |
| A saved creature behaves differently | Check backend, bounds, rules, velocities, neighboring particles, audio modulation, and grid setting. |
| A session loads without music | Audio is not embedded or autoplayed. Locate the referenced file and start playback. |
| Build fails because Rust is too old | With a rustup installation, run `rustup update stable`, then retry. Keep the included lockfile. |

For a crash report, launch with logging enabled and keep the terminal output.

**PowerShell:**

```powershell
$env:RUST_LOG = "info"
$env:RUST_BACKTRACE = "1"
cargo run --release --locked
```

**macOS / Linux:**

```sh
RUST_LOG=info RUST_BACKTRACE=1 cargo run --release --locked
```

Include your OS, GPU, `rustc --version`, the error output, and the steps that triggered the issue.

## Build and validation notes

Normal checks:

```sh
cargo test --locked
cargo build --release --locked
```

The additional graphics tests require a compatible graphics or software adapter:

```sh
cargo test --locked -- --include-ignored
```

The previous 0.3.0 validation completed a release build and all 22 tests using Linux software Vulkan. This is not a native Windows/macOS compatibility certification or a hardware performance benchmark. See the source archive's `VALIDATION.md` for the recorded scope and `CHANGELOG.md` for implementation changes.

This guide documents the existing 0.3.0 application; it does not change its code. The project is licensed under MIT; see `LICENSE` in the source package.
