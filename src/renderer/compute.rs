use wgpu::util::DeviceExt;
use crate::sim::{SimState, Particle};

// GPU-side params uniform — must match particle_physics.wgsl Params struct
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    particle_count: u32,
    type_count:     u32,
    r_max:          f32,
    force_scale:    f32,
    friction:       f32,
    dt:             f32,
    bounds:         f32,
    wrap:           u32,
}

impl GpuParams {
    fn from_sim(sim: &SimState) -> Self {
        Self {
            particle_count: sim.particles.len() as u32,
            type_count:     sim.params.type_count as u32,
            r_max:          sim.params.r_max,
            force_scale:    sim.params.force_scale,
            friction:       sim.params.friction,
            dt:             sim.params.dt,
            bounds:         sim.params.bounds,
            wrap:           if sim.params.wrap { 1 } else { 0 },
        }
    }
}

pub struct ComputePipeline {
    pipeline:       wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    // GPU buffers — ping-pong between frames
    pub buf_a:      wgpu::Buffer,  // particles in
    pub buf_b:      wgpu::Buffer,  // particles out
    buf_params:     wgpu::Buffer,
    buf_forces:     wgpu::Buffer,

    capacity:       usize,         // max particles this pipeline was built for
    pub ping:       bool,          // which buffer is currently "in"
}

impl ComputePipeline {
    pub fn new(device: &wgpu::Device, initial_capacity: usize) -> Self {
        // Load and compile the WGSL shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_physics"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/particle_physics.wgsl").into()
            ),
        });

        // Describe what bindings the shader expects
        let bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bgl"),
                entries: &[
                    // binding 0: params uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: particles_in (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: particles_out (read-write storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 3: force_matrix (read-only storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pl"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("particle_physics"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            }
        );

        // Allocate GPU buffers at initial capacity
        let particle_bytes = initial_capacity * std::mem::size_of::<Particle>();
        let buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_a"),
            size: particle_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_b"),
            size: particle_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Force matrix max: 32 types × 32 types × 4 bytes
        let buf_forces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("force_matrix"),
            size: (32 * 32 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            buf_a,
            buf_b,
            buf_params,
            buf_forces,
            capacity: initial_capacity,
            ping: true,
        }
    }

    /// Upload CPU sim state to GPU and dispatch the compute shader.
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sim: &mut SimState) {
        let count = sim.particles.len();
        if count == 0 { return; }

        // Re-allocate buffers if we've exceeded capacity
        if count > self.capacity {
            *self = Self::new(device, count * 2);
            sim.particles_dirty = true;
            sim.force_matrix_dirty = true;
            sim.params_dirty = true;
        }

        // Upload particle data if dirty
        if sim.particles_dirty {
            let data = bytemuck::cast_slice(&sim.particles);
            let (buf_in, _) = self.bufs();
            queue.write_buffer(buf_in, 0, data);
        }

        // Upload force matrix if dirty
        if sim.force_matrix_dirty {
            let data = bytemuck::cast_slice(&sim.force_matrix);
            queue.write_buffer(&self.buf_forces, 0, data);
        }

        // Always upload params (cheap)
        let gpu_params = GpuParams::from_sim(sim);
        queue.write_buffer(&self.buf_params, 0, bytemuck::bytes_of(&gpu_params));

        sim.clear_dirty();

        // Build bind group — wires buffers to shader bindings
        let (buf_in, buf_out) = self.bufs();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_in.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_forces.as_entire_binding() },
            ],
        });

        // Dispatch
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("compute") }
        );
        {
            let mut pass = encoder.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: Some("physics"), timestamp_writes: None }
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Each workgroup handles 64 particles
            let workgroups = (count as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Flip ping-pong: output becomes input next frame
        self.ping = !self.ping;
    }

    /// Returns (buf_in, buf_out) based on current ping state
    fn bufs(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        if self.ping { (&self.buf_a, &self.buf_b) }
        else         { (&self.buf_b, &self.buf_a) }
    }

    /// The buffer that holds the latest particle positions (used by draw pipeline)
    pub fn current_buf(&self) -> &wgpu::Buffer {
        // After run(), ping has been flipped, so current output is the old input side
        if self.ping { &self.buf_a } else { &self.buf_b }
    }
}
