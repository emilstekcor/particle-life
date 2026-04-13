use wgpu::util::DeviceExt;
use egui_wgpu::ScreenDescriptor;
use glam::{Vec3, Mat4};
use crate::sim::{SimState, Particle};
use crate::ui::UiState;
use super::compute::ComputePipeline;

// Vertex format for particle rendering
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn new(pos: Vec3, kind: u32) -> Self {
        Self {
            position: pos.into(),
            color: type_color(kind),
        }
    }
}

// Camera uniforms
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    _padding: f32,
}

pub struct DrawPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    vertex_buffer: wgpu::Buffer,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl DrawPipeline {
    pub fn new(device: &wgpu::Device, surface_cfg: &wgpu::SurfaceConfiguration) -> Self {
        // Vertex shader - simple pass-through
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vertex.wgsl").into()),
        });

        // Fragment shader - solid color
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fragment.wgsl").into()),
        });

        // Bind group layout for camera uniforms
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Bind Group Layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_cfg.format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Camera buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
            ],
        });

        // Vertex buffer (will be updated each frame)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: (10000 * std::mem::size_of::<Vertex>()) as u64, // Max 10k particles
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bind_group_layout,
            vertex_buffer,
            camera_buffer,
            camera_bind_group,
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        compute: &ComputePipeline,
        sim: &SimState,
        ui: &UiState,
        screen_desc: &ScreenDescriptor,
        queue: &wgpu::Queue,
    ) {
        // Update camera uniforms
        let camera_uniform = self.create_camera_uniform(sim, ui, screen_desc);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Create vertex buffer from particle positions
        if !sim.particles.is_empty() {
            let vertices: Vec<Vertex> = sim.particles
                .iter()
                .map(|p| Vertex::new(Vec3::from(p.position), p.kind))
                .collect();

            queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&vertices),
            );
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.02,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

        if !sim.particles.is_empty() {
            render_pass.draw(0..sim.particles.len() as u32, 0..1);
        }
    }

    fn create_camera_uniform(&self, sim: &SimState, ui: &UiState, screen_desc: &ScreenDescriptor) -> CameraUniform {
        // Create orbit camera
        let center = Vec3::splat(sim.params.bounds * 0.5);
        
        let camera_pos = center + Vec3::new(
            ui.camera_dist * ui.camera_yaw.cos() * ui.camera_pitch.cos(),
            ui.camera_dist * ui.camera_pitch.sin(),
            ui.camera_dist * ui.camera_yaw.sin() * ui.camera_pitch.cos(),
        );

        let view = Mat4::look_at_rh(camera_pos, center, Vec3::Y);
        
        // Simple orthographic projection
        let aspect = screen_desc.size_in_pixels[0] as f32 / screen_desc.size_in_pixels[1] as f32;
        let proj = Mat4::orthographic_rh(
            -sim.params.bounds * 0.5 * aspect,
            sim.params.bounds * 0.5 * aspect,
            -sim.params.bounds * 0.5,
            sim.params.bounds * 0.5,
            0.1,
            10.0,
        );

        let view_proj = proj * view;

        CameraUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: camera_pos.to_array(),
            _padding: 0.0,
        }
    }
}

// Helper function to get particle type colors
fn type_color(kind: u32) -> [f32; 3] {
    let colors = [
        [0.96, 0.36, 0.36], // red
        [0.36, 0.76, 0.96], // blue  
        [0.56, 0.96, 0.36], // green
        [0.96, 0.76, 0.26], // yellow
        [0.86, 0.46, 0.96], // purple
        [0.96, 0.56, 0.26], // orange
        [0.36, 0.96, 0.76], // cyan
        [0.96, 0.76, 0.86], // pink
    ];
    colors[kind as usize % colors.len()]
}
