I feel it merits mentioning that I used ai for massive portions of this project specifically in porting over my already existing code from c++. I frankly did't have the time to figure out prarallel processing syntaxes in rust and just had an ai implement the build features in what I had made.I consider it a miracle it works and I will keep this as a notice and warning.  



# Particle Life 3D

A Rust/WGPU particle-life sandbox for experimenting with emergent 3D structures, force rules, and reaction-driven behavior.

The long-term goal is to build toward artificial-life style systems where particles can form stable structures and use reaction/probability rules as a simple compute layer. Eventually, I want to test whether small neural-like control systems can emerge from or be mapped onto these particle structures, then trained inside an environment.

Current features:
- CPU/GPU physics modes
- Type-based force matrix
- Particle reactions
- Trail rendering
- Real-time egui controls

Planned direction:
- More stable 3D structures
- Reaction-based behavior logic
- Probability-biased movement
- Small trainable agent/neuron experiments

## Requirements

- Rust / Cargo
- Working GPU drivers
- Vulkan, DX12, or Metal-compatible GPU through `wgpu`

Linux users should have Vulkan drivers installed.

## Run

```bash
cargo run --release
