use egui_wgpu::ScreenDescriptor;
use glam::{Vec3, Mat4};
use crate::ui::UiState;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color:    [f32; 3],
}


impl Vertex {
    fn new(pos: Vec3, kind: u32) -> Self {
        let colors = [
            [0.96, 0.36, 0.36],
            [0.36, 0.76, 0.96],
            [0.56, 0.96, 0.36],
            [0.96, 0.76, 0.26],
            [0.86, 0.46, 0.96],
            [0.96, 0.56, 0.26],
            [0.36, 0.96, 0.76],
            [0.96, 0.76, 0.86],
        ];
        Self { position: pos.into(), color: colors[kind as usize % colors.len()] }
    }
    
    fn with_color(pos: Vec3, color: [f32; 3]) -> Self {
        Self { position: pos.into(), color }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj:  [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding:   f32,
}

pub struct DrawPipeline {
    pipeline:          wgpu::RenderPipeline,
    trail_line_pipeline: wgpu::RenderPipeline,
    trail_point_pipeline: wgpu::RenderPipeline,
    pub camera_buffer:     wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    selection_bind_group: wgpu::BindGroup,
    trail_bind_group_layout: wgpu::BindGroupLayout,
    trail_bind_group: wgpu::BindGroup,
}

impl DrawPipeline {
    pub fn new(device: &wgpu::Device, surface_cfg: &wgpu::SurfaceConfiguration) -> Self {
        let vert = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vertex.wgsl").into()),
        });
        let frag = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fragment.wgsl").into()),
        });

        // bind_group_layout is only needed here; not stored as a field
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let selection_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Render Pipeline Layout"),
            bind_group_layouts:   &[&bind_group_layout, &selection_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:      &vert,
                entry_point: "main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<crate::sim::GpuParticle>() as wgpu::BufferAddress,
                    step_mode:    wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset:           0,
                            shader_location:  0,
                            format:           wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset:          12,
                            shader_location: 1,
                            format:          wgpu::VertexFormat::Uint32,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module:      &frag,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format:     surface_cfg.format,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:           wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face:         wgpu::FrontFace::Ccw,
                cull_mode:          None,
                polygon_mode:       wgpu::PolygonMode::Fill,
                unclipped_depth:    false,
                conservative:       false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count:                     1,
                mask:                      !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        
        // Create trail shaders
        let trail_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Trail Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/trail_line.wgsl").into()),
        });

        let trail_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Trail BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let trail_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Trail Pipeline Layout"),
            bind_group_layouts: &[&trail_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create line trail pipeline
        let trail_line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Trail Line Pipeline"),
            layout: Some(&trail_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &trail_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &trail_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_cfg.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create point trail pipeline
        let trail_point_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Trail Point Pipeline"),
            layout: Some(&trail_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &trail_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &trail_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_cfg.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Camera Buffer"),
            size:               std::mem::size_of::<CameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Camera Bind Group"),
            layout:  &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Selection bind group will be created later when we have the selection buffer
        let selection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Selection Bind Group"),
            layout:  &selection_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Dummy Selection Buffer"),
                        size: 4,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // Trail bind group will be created later when we have the trail buffers
        let trail_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Trail Bind Group"),
            layout: &trail_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Dummy Trail History Buffer"),
                            size: 4,
                            usage: wgpu::BufferUsages::STORAGE,
                            mapped_at_creation: false,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("Dummy Trail Params Buffer"),
                            size: 4,
                            usage: wgpu::BufferUsages::UNIFORM,
                            mapped_at_creation: false,
                        }),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        Self { 
            pipeline, 
            trail_line_pipeline,
            trail_point_pipeline,
            camera_buffer, 
            camera_bind_group,
            selection_bind_group,
            trail_bind_group_layout,
            trail_bind_group,
        }
    }

    pub fn update_selection_bind_group(&mut self, device: &wgpu::Device, selection_buf: &wgpu::Buffer) {
        let selection_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Selection BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        self.selection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Selection Bind Group"),
            layout:  &selection_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: selection_buf.as_entire_binding(),
            }],
        });
    }

    pub fn update_trail_bind_group(
        &mut self,
        device: &wgpu::Device,
        trail_history_buf: &wgpu::Buffer,
        trail_params_buf: &wgpu::Buffer,
    ) {
        self.trail_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Trail Bind Group"),
            layout: &self.trail_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buffer.as_entire_binding(),
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
    }

    pub fn render(
        &mut self,
        encoder:       &mut wgpu::CommandEncoder,
        _view:         &wgpu::TextureView,
        depth_view:    &wgpu::TextureView,
        particle_buf:  &wgpu::Buffer,
        particle_count: u32,
        ui:            &mut UiState,
        screen_desc:   &ScreenDescriptor,
        queue:         &wgpu::Queue,
        trail_valid_len: u32,
        load_op:       wgpu::LoadOp<wgpu::Color>,
        particle_target: &wgpu::TextureView,  // Target for particle rendering (trace texture or swapchain)
    ) {
        let (uniform, view_matrix, view_proj) = self.build_camera(ui, screen_desc);
        ui.view_matrix = view_matrix;
        ui.view_proj   = view_proj;
        ui.viewport    = screen_desc.size_in_pixels;
        // Do not override ui.slice_center here; it is user-controlled from the UI.

        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // Particles are now uploaded by the compute pipeline - no need to upload here

        // GPU selection handles highlighting - no CPU overlay needed

        
        // Particle pass - use load_op passed from renderer
        // This allows proper trace accumulation using persistent texture

        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Particle Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: particle_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  load_op,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes:         None,
            occlusion_query_set:      None,
        });

        // Draw trails based on render mode
        match ui.trace_render_mode {
            crate::ui::TraceRenderMode::Off => {}
            crate::ui::TraceRenderMode::Simple => {} // Removed - no longer used
            crate::ui::TraceRenderMode::Lines => {
                if particle_count > 0 && trail_valid_len >= 2 {
                    // Debug mode: cap to first 32 particles for easier inspection
                    let debug_cap = false; // Set to true for easier visual testing
                    let test_particle_count = if debug_cap { particle_count.min(32) } else { particle_count };
                    
                    // Calculate trail vertices for line segments
                    let valid_segments = trail_valid_len - 1;
                    let trail_vertex_count = test_particle_count * valid_segments * 2;
                    
                    if trail_vertex_count > 0 {
                        rp.set_pipeline(&self.trail_line_pipeline);
                        rp.set_bind_group(0, &self.trail_bind_group, &[]);
                        rp.draw(0..trail_vertex_count, 0..1);
                    }
                }
            }
            crate::ui::TraceRenderMode::Dots => {
                if particle_count > 0 && trail_valid_len >= 1 {
                    // Debug mode: cap to first 32 particles for easier inspection
                    let debug_cap = false; // Set to true for easier visual testing
                    let test_particle_count = if debug_cap { particle_count.min(32) } else { particle_count };
                    
                    // Calculate trail vertices - each history sample becomes one point
                    let trail_vertex_count = test_particle_count * trail_valid_len;
                    
                    if trail_vertex_count > 0 {
                        rp.set_pipeline(&self.trail_point_pipeline);
                        rp.set_bind_group(0, &self.trail_bind_group, &[]);
                        rp.draw(0..trail_vertex_count, 0..1);
                    }
                }
            }
        }

        // Draw particles on top
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, &self.camera_bind_group, &[]);
        rp.set_bind_group(1, &self.selection_bind_group, &[]);
        rp.set_vertex_buffer(0, particle_buf.slice(..));

        if particle_count > 0 {
            rp.draw(0..particle_count, 0..1);
        }
    }

    fn build_camera(
        &self,
        ui:          &UiState,
        screen_desc: &ScreenDescriptor,
    ) -> (CameraUniform, Mat4, Mat4) {
        let pos    = ui.fly_pos;
        let target = pos + ui.fly_forward();
        let view   = Mat4::look_at_rh(pos, target, Vec3::Y);

        let aspect = screen_desc.size_in_pixels[0] as f32
                   / screen_desc.size_in_pixels[1] as f32;

        let proj = Mat4::perspective_rh(
            60_f32.to_radians(),
            aspect,
            0.001,   // near — can get very close to particles
            500.0,   // far  — see the whole sim from distance
        );

        let view_proj = proj * view;
        let uniform   = CameraUniform {
            view_proj:  view_proj.to_cols_array_2d(),
            camera_pos: pos.to_array(),
            _padding:   0.0,
        };
        (uniform, view, view_proj)
    }
}
