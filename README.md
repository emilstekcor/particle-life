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
