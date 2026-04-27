I had no time to work on this properly, but I made the basis for this while learning rust then adapted code I wrote to this already in opengl using claude.It has not been checked nor tested and should serve as a basis. With that said it is my favorite sim so far. I used a 3060 to run it and I have had fun, make sure the gpu mode is toggled on. 



# Particle Life 3D

A Rust/WGPU particle-life sandbox for experimenting with emergent behavior in 3D.  
It supports CPU/GPU physics modes, type-based force rules, particle reactions, trail rendering, and a real-time egui control panel.

## Requirements

- Rust / Cargo
- A GPU with Vulkan, Metal, DX12, or WebGPU-compatible support through `wgpu`
- Linux, Windows, or macOS

## Run

```bash
cargo run --release
