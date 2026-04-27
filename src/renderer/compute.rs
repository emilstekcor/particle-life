use crate::sim::{GpuParticle, SimParams};
use std::mem;

pub const MAX_TYPES: usize = 32;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParams {
    pub dt: f32,
    pub r_max: f32,
    pub force_scale: f32,
    pub friction: f32,
    pub beta: f32,
    pub bounds: f32,
    pub max_speed: f32,
    pub type_count: u32,
    pub count: u32,
    pub wrap: u32,
    pub reactions_enabled: u32,
    pub mix_radius: f32,
    pub reaction_probability: f32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SelectionParams {
    pub mode: u32,
    pub _pad0: [u32; 3], // align to 16 bytes

    pub rect_min: [f32; 2],
    pub _pad1: [f32; 2],

    pub rect_max: [f32; 2],
    pub _pad2: [f32; 2],

    pub brush_center: [f32; 2],
    pub _pad3: [f32; 2],

    pub brush_radius: f32,
    pub slice_depth: f32,
    pub _pad4: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrailPoint {
    pub position: [f32; 3],
    pub kind: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrailParams {
    pub particle_count: u32,
    pub trail_len: u32,
    pub head: u32,
    pub valid_len: u32,
    pub enabled: u32,
    pub type_filter: i32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl From<&SimParams> for GpuParams {
    fn from(params: &SimParams) -> Self {
        Self {
            dt: params.dt,
            r_max: params.r_max,
            force_scale: params.force_scale,
            friction: params.friction,
            beta: params.beta,
            bounds: params.bounds,
            max_speed: params.max_speed,
            type_count: params.type_count as u32,
            count: 0, // Must be set explicitly based on particle count
            wrap: if params.wrap { 1 } else { 0 },
            reactions_enabled: if params.reactions_enabled { 1 } else { 0 },
            mix_radius: params.mix_radius,
            reaction_probability: params.reaction_probability,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}

pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub compute_bind_groups: [wgpu::BindGroup; 2],

    pub particle_buffers: [wgpu::Buffer; 2], // ping-pong particle buffers
    pub particle_read_index: usize,
    pub particle_write_index: usize,

    params_buf: wgpu::Buffer,
    rules_buf: wgpu::Buffer,
    reaction_buf: wgpu::Buffer,      // reaction table (i32 per type pair)
    pub trace_len_buf: wgpu::Buffer, // trace length matrix (u32 per type pair)
    pub trace_timer_buf: wgpu::Buffer, // trace timers (u32 per particle)
    pub trace_prev_pos_buf: wgpu::Buffer, // trace previous positions (vec3 per particle)
    pub selection_buf: wgpu::Buffer, // selection flags (1 byte per particle)
    selection_params_buf: wgpu::Buffer, // selection parameters
    readback_buf: wgpu::Buffer,      // staging buffer for GPU->CPU readback

    // Trail system
    pub trail_history_buf: wgpu::Buffer,
    pub trail_params_buf: wgpu::Buffer,
    trail_capture_pipeline: wgpu::ComputePipeline,
    trail_capture_bind_group_layout: wgpu::BindGroupLayout,
    trail_capture_bind_group: wgpu::BindGroup,

    pub trail_len: u32,
    pub trail_head: u32,
    pub trail_valid_len: u32,
    pub trail_type_filter: i32,
    pub trails_enabled: bool,

    pub particle_count: u32,
}

impl ComputePipeline {
    pub fn new(device: &wgpu::Device, max_particles: usize) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Particle buffer read (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Particle buffer write (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params buffer (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Rules buffer (storage, read)
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
                // Selection flags buffer (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Selection parameters buffer (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Reaction table buffer (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Trace length matrix buffer (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Trace timer buffer (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Trace previous position buffer (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Create two particle buffers for ping-pong
        let particle_buffers = [
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Buffer A"),
                size: (max_particles * mem::size_of::<GpuParticle>()) as u64,
                usage: wgpu::BufferUsages::STORAGE    // compute reads/writes
                     | wgpu::BufferUsages::VERTEX     // render reads
                     | wgpu::BufferUsages::COPY_DST   // initial upload
                     | wgpu::BufferUsages::COPY_SRC, // GPU->CPU readback
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Buffer B"),
                size: (max_particles * mem::size_of::<GpuParticle>()) as u64,
                usage: wgpu::BufferUsages::STORAGE    // compute reads/writes
                     | wgpu::BufferUsages::VERTEX     // render reads
                     | wgpu::BufferUsages::COPY_DST   // initial upload
                     | wgpu::BufferUsages::COPY_SRC, // GPU->CPU readback
                mapped_at_creation: false,
            }),
        ];

        // Create params buffer
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create rules buffer (max MAX_TYPES×MAX_TYPES)
        let rules_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rules Buffer"),
            size: (MAX_TYPES * MAX_TYPES * mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create reaction buffer (max MAX_TYPES×MAX_TYPES)
        let reaction_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reaction Buffer"),
            size: (MAX_TYPES * MAX_TYPES * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create trace length buffer (max MAX_TYPES×MAX_TYPES)
        let trace_len_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trace Length Buffer"),
            size: (MAX_TYPES * MAX_TYPES * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create trace timer buffer (u32 per particle)
        let trace_timer_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trace Timer Buffer"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create trace previous position buffer (vec3 per particle)
        let trace_prev_pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trace Prev Position Buffer"),
            size: (max_particles * std::mem::size_of::<[f32; 3]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create selection flags buffer (u32 per particle for proper WGSL alignment)
        let selection_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selection Buffer"),
            size: (max_particles * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create selection parameters buffer
        let selection_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selection Params Buffer"),
            size: mem::size_of::<SelectionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create readback buffer for GPU->CPU sync (CPU visible)
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: (max_particles * mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create trail buffers
        const DEFAULT_TRAIL_LEN: u32 = 16;
        let trail_len = DEFAULT_TRAIL_LEN;
        let trail_history_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trail History Buffer"),
            size: (max_particles as u64)
                * (trail_len as u64)
                * (std::mem::size_of::<TrailPoint>() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let trail_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trail Params Buffer"),
            size: std::mem::size_of::<TrailParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create trail capture pipeline
        let trail_capture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Trail Capture Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/trail_capture.wgsl").into()),
        });

        let trail_capture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Trail Capture BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let trail_capture_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Trail Capture Pipeline Layout"),
                bind_group_layouts: &[&trail_capture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let trail_capture_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Trail Capture Pipeline"),
                layout: Some(&trail_capture_pipeline_layout),
                module: &trail_capture_shader,
                entry_point: "main",
            });

        let trail_capture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Trail Capture Bind Group"),
            layout: &trail_capture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffers[0].as_entire_binding(), // Will be updated dynamically
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: trail_history_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: trail_params_buf.as_entire_binding(),
                },
            ],
        });

        // Create two compute bind groups for ping-pong
        let compute_bind_groups = [
            // Bind group 0: reads from buffer[0], writes to buffer[1]
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group 0"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: rules_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: selection_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: selection_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: reaction_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: trace_len_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: trace_timer_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: trace_prev_pos_buf.as_entire_binding(),
                    },
                ],
            }),
            // Bind group 1: reads from buffer[1], writes to buffer[0]
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group 1"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_buffers[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: rules_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: selection_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: selection_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: reaction_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: trace_len_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: trace_timer_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: trace_prev_pos_buf.as_entire_binding(),
                    },
                ],
            }),
        ];

        Self {
            pipeline,
            bind_group_layout,
            compute_bind_groups,
            particle_buffers,
            particle_read_index: 0,
            particle_write_index: 1,
            params_buf,
            rules_buf,
            reaction_buf,
            trace_len_buf,
            trace_timer_buf,
            trace_prev_pos_buf,
            selection_buf,
            selection_params_buf,
            readback_buf,
            trail_history_buf,
            trail_params_buf,
            trail_capture_pipeline,
            trail_capture_bind_group_layout,
            trail_capture_bind_group,
            trail_len: 16,
            trail_head: 0,
            trail_valid_len: 0,
            trail_type_filter: -1,
            trails_enabled: false,
            particle_count: 0,
        }
    }

    pub fn current_render_particle_buffer(&self) -> &wgpu::Buffer {
        &self.particle_buffers[self.particle_read_index]
    }

    pub fn current_compute_bind_group(&self) -> &wgpu::BindGroup {
        &self.compute_bind_groups[self.particle_read_index]
    }

    pub fn swap_particle_buffers(&mut self) {
        std::mem::swap(
            &mut self.particle_read_index,
            &mut self.particle_write_index,
        );
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, particle_count: u32) {
        let workgroups = (particle_count + 63) / 64; // 64 threads per workgroup
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Physics Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, self.current_compute_bind_group(), &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn upload_particles(&mut self, queue: &wgpu::Queue, particles: &[GpuParticle]) {
        // Write to both particle buffers to avoid stale data
        queue.write_buffer(
            &self.particle_buffers[0],
            0,
            bytemuck::cast_slice(particles),
        );
        queue.write_buffer(
            &self.particle_buffers[1],
            0,
            bytemuck::cast_slice(particles),
        );
        self.particle_count = particles.len() as u32;
        // Reset indices after upload
        self.particle_read_index = 0;
        self.particle_write_index = 1;
    }

    pub fn upload_params(&self, queue: &wgpu::Queue, params: &GpuParams) {
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(params));
    }

    pub fn upload_rules(&self, queue: &wgpu::Queue, rules: &[f32]) {
        let max_bytes = MAX_TYPES * MAX_TYPES * std::mem::size_of::<f32>();
        let upload_bytes = std::mem::size_of_val(rules);

        assert!(
            upload_bytes <= max_bytes,
            "rules buffer overflow: trying to upload {} bytes, capacity {} bytes ({} types)",
            upload_bytes,
            max_bytes,
            MAX_TYPES
        );

        queue.write_buffer(&self.rules_buf, 0, bytemuck::cast_slice(rules));
    }

    pub fn upload_reactions(&self, queue: &wgpu::Queue, table: &[i32]) {
        queue.write_buffer(&self.reaction_buf, 0, bytemuck::cast_slice(table));
    }

    pub fn upload_trace_lengths(&self, queue: &wgpu::Queue, trace_len_matrix: &[u32]) {
        let max_bytes = MAX_TYPES * MAX_TYPES * std::mem::size_of::<u32>();
        let upload_bytes = std::mem::size_of_val(trace_len_matrix);

        assert!(
            upload_bytes <= max_bytes,
            "trace_len_matrix buffer overflow: trying to upload {} bytes, capacity {} bytes ({} types)",
            upload_bytes,
            max_bytes,
            MAX_TYPES
        );

        queue.write_buffer(
            &self.trace_len_buf,
            0,
            bytemuck::cast_slice(trace_len_matrix),
        );
    }

    pub fn upload_selection_params(&self, queue: &wgpu::Queue, params: &SelectionParams) {
        queue.write_buffer(&self.selection_params_buf, 0, bytemuck::bytes_of(params));
    }

    pub fn clear_selection(&self, queue: &wgpu::Queue, particle_count: u32) {
        // Clear selection flags to 0 using u32 values
        let zero_data = vec![0u32; particle_count as usize];
        queue.write_buffer(&self.selection_buf, 0, bytemuck::cast_slice(&zero_data));
    }

    /// Read back selection flags and convert to selected_indices
    pub fn readback_selection(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
    ) -> Result<Vec<usize>, wgpu::BufferAsyncError> {
        if particle_count == 0 {
            return Ok(Vec::new());
        }

        // Create a staging buffer for this readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selection Readback Staging"),
            size: (particle_count as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create a temporary command encoder for the copy
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Selection Readback Copy"),
        });

        // Copy from selection buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.selection_buf,
            0,
            &staging_buffer,
            0,
            (particle_count as usize * std::mem::size_of::<u32>()) as u64,
        );

        // Submit the command and wait for completion
        let command_buffer = encoder.finish();
        queue.submit([command_buffer]);
        device.poll(wgpu::Maintain::Wait);

        // Map the staging buffer synchronously (blocking but functional)
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for the mapping to complete
        rx.recv().unwrap()?;

        // Read the selection data
        let selection_data = buffer_slice.get_mapped_range();
        let selection_flags: &[u32] = bytemuck::cast_slice(&selection_data);

        // Convert selection flags to indices
        let mut selected_indices = Vec::new();
        for (i, &flag) in selection_flags.iter().enumerate() {
            if flag != 0 {
                selected_indices.push(i);
            }
        }

        // Unmap and drop the staging buffer
        drop(selection_data);
        staging_buffer.unmap();
        drop(staging_buffer);

        Ok(selected_indices)
    }

    pub fn upload_trail_params(&self, queue: &wgpu::Queue) {
        let params = TrailParams {
            particle_count: self.particle_count,
            trail_len: self.trail_len,
            head: self.trail_head,
            valid_len: self.trail_valid_len,
            enabled: self.trails_enabled as u32,
            type_filter: self.trail_type_filter,
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.trail_params_buf, 0, bytemuck::bytes_of(&params));
    }

    pub fn dispatch_trail_capture(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.trails_enabled || self.particle_count == 0 {
            return;
        }

        let workgroups = (self.particle_count + 63) / 64;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Trail Capture Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.trail_capture_pipeline);
        cpass.set_bind_group(0, &self.trail_capture_bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub fn advance_trail_head(&mut self) {
        if !self.trails_enabled {
            return;
        }
        self.trail_head = (self.trail_head + 1) % self.trail_len;
        if self.trail_valid_len < self.trail_len {
            self.trail_valid_len += 1;
        }
    }

    pub fn reset_trails(&mut self) {
        self.trail_head = 0;
        self.trail_valid_len = 0;
    }

    pub fn clear_trail_history(&self, encoder: &mut wgpu::CommandEncoder, max_particles: usize) {
        let size =
            (max_particles * (self.trail_len as usize) * std::mem::size_of::<TrailPoint>()) as u64;

        // Use efficient buffer clearing instead of allocating a huge zero Vec
        encoder.clear_buffer(&self.trail_history_buf, 0, Some(size));
    }
}
