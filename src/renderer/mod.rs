pub mod draw;
pub mod compute;

use winit::window::Window;
use winit::event::Event;
use egui_wgpu::ScreenDescriptor;

use crate::sim::SimState;
use crate::ui::UiState;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct DepthTexture {
    _texture: wgpu::Texture,
    view:     wgpu::TextureView,
}

impl DepthTexture {
    fn create(device: &wgpu::Device, cfg: &wgpu::SurfaceConfiguration) -> Self {
        let size = wgpu::Extent3d { 
            width: cfg.width.max(1), 
            height: cfg.height.max(1), 
            depth_or_array_layers: 1 
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { _texture: texture, view }
    }
}

/// Owns the wgpu device/queue/surface and all GPU pipelines.
pub struct Renderer {
    // Core wgpu handles
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    surface:     wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    depth:       DepthTexture,

    
    // Compute pipeline for GPU physics
    pub compute: compute::ComputePipeline,

    // Render pipeline for drawing particles as points/billboards
    pub draw: draw::DrawPipeline,

    // egui integration
    egui_ctx:      egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    egui_state:    egui_winit::State,
}

impl Renderer {
    /// Sync GPU selection with CPU selected_indices
    pub fn sync_selection(&mut self, ui: &mut UiState) {
        if ui.selection_readback_needed {
            match self.compute.readback_selection(&self.device, &self.queue, self.compute.particle_count) {
                Ok(selected_indices) => {
                    ui.selected_indices = selected_indices;
                    log::debug!("GPU selection synced: {} particles selected", ui.selected_indices.len());
                }
                Err(e) => {
                    log::warn!("Failed to read back selection: {:?}", e);
                }
            }
            ui.selection_readback_needed = false; // Reset flag
        }
    }

    pub async fn new(window: &Window) -> Self {
        // ── wgpu init ──────────────────────────────────────────────────────────
        // Instance = entry point into wgpu. Auto-picks Vulkan/Metal/DX12/WebGPU.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // Safety: surface must not outlive the window.
        // We use 'static here and guarantee the window lives for the program.
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(window).unwrap()
            )
        }.unwrap();

        // Adapter = handle to a physical GPU
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        log::info!("GPU: {:?}", adapter.get_info().name);

        // Device = logical GPU handle. Queue = command submission queue.
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ).await.unwrap();

        // Surface config — sets pixel format, size, vsync
        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let format = surface_caps.formats[0]; // pick first supported format

        let surface_cfg = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_cfg);
        let depth = DepthTexture::create(&device, &surface_cfg);

        
        // ── egui init ──────────────────────────────────────────────────────────
        let egui_ctx = egui::Context::default();
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window,
            None,
            None,
        );

        // ── GPU pipelines ──────────────────────────────────────────────────────
        let compute = compute::ComputePipeline::new(&device, crate::sim::MAX_RENDER_PARTICLES);
        let mut draw = draw::DrawPipeline::new(&device, &surface_cfg);
        
        // Initialize bind groups with actual buffers
        draw.update_selection_bind_group(&device, &compute.selection_buf);
        draw.update_trail_bind_group(&device, &compute.trail_history_buf, &compute.trail_params_buf);

        Self {
            device,
            queue,
            surface,
            surface_cfg,
            depth,
            compute,
            draw,
            egui_ctx,
            egui_renderer,
            egui_state,
        }
    }

    /// Forward window events to egui. Returns EventResponse with consumed/repaint flags.
    pub fn egui_handle_event(&mut self, window: &Window, event: &Event<()>) -> egui_winit::EventResponse {
        if let winit::event::Event::WindowEvent { event, .. } = event {
            self.egui_state.on_window_event(window, event)
        } else {
            egui_winit::EventResponse::default()
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 { return; }
        self.surface_cfg.width = size.width;
        self.surface_cfg.height = size.height;
        self.surface.configure(&self.device, &self.surface_cfg);
        self.depth = DepthTexture::create(&self.device, &self.surface_cfg);
    }

    /// Main render function called every frame.
    pub fn render(&mut self, window: &Window, sim: &mut SimState, ui: &mut UiState) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.surface_cfg);
                return;
            }
            Err(_) => return,
        };

        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Frame") }
        );

        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        // 1) Run egui first so ui state is current for this frame
        let raw_input = self.egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            crate::ui::draw_ui(ctx, sim, ui);
        });

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        // 2) Upload dirty simulation data after UI may have changed it
        if sim.particles_dirty {
            let gpu_particles: Vec<crate::sim::GpuParticle> = sim.particles.iter()
                .map(crate::sim::GpuParticle::from_particle)
                .collect();

            let particle_count = gpu_particles.len() as u32;
            let old_particle_count = self.compute.particle_count;
            self.compute.upload_particles(&self.queue, &gpu_particles);

            // Reset trails if all particles are cleared
            if particle_count == 0 && old_particle_count > 0 {
                self.compute.reset_trails();
                let temp_encoder = &mut encoder;
                self.compute.clear_trail_history(temp_encoder, crate::sim::MAX_RENDER_PARTICLES);
            }

            let mut gpu_params = compute::GpuParams::from(&sim.params);
            gpu_params.count = particle_count;
            self.compute.upload_params(&self.queue, &gpu_params);
            self.compute.upload_rules(&self.queue, &sim.force_matrix);
            self.compute.upload_reactions(&self.queue, &sim.reaction_table);

            sim.particles_dirty = false;
            sim.params_dirty = false;
            sim.force_matrix_dirty = false;
        } else if sim.params_dirty || sim.force_matrix_dirty {
            let mut gpu_params = compute::GpuParams::from(&sim.params);
            gpu_params.count = self.compute.particle_count;
            self.compute.upload_params(&self.queue, &gpu_params);
            self.compute.upload_rules(&self.queue, &sim.force_matrix);
            self.compute.upload_reactions(&self.queue, &sim.reaction_table);

            sim.params_dirty = false;
            sim.force_matrix_dirty = false;
        }

        // Upload GPU selection parameters
        self.compute.upload_selection_params(&self.queue, &ui.gpu_selection_params);

        // Do not clear every frame. This is expensive at high particle counts.
        // Clear only when selection mode changes, particles are reset, or user presses Clear Selection.
        // if ui.gpu_selection_params.mode == 0 {
        //     self.compute.clear_selection(&self.queue, self.compute.particle_count);
        // }

        // 3) Sync UI state into compute trail parameters
        // Clear trail history when enabling trails mid-sim to avoid garbage
        let trails_enabled = ui.trace_render_mode != crate::ui::TraceRenderMode::Off;
        let trails_newly_enabled = trails_enabled && !self.compute.trails_enabled;
        if trails_newly_enabled {
            self.compute.reset_trails();
            self.compute.clear_trail_history(&mut encoder, crate::sim::MAX_RENDER_PARTICLES);
        }
        
        self.compute.trails_enabled = trails_enabled;
        let trail_len_changed = ui.trace_len != self.compute.trail_len;
        self.compute.trail_len = ui.trace_len.clamp(1, 16);
        if trail_len_changed {
            self.compute.reset_trails();
            self.compute.clear_trail_history(&mut encoder, crate::sim::MAX_RENDER_PARTICLES);
        }
        self.compute.trail_type_filter = ui.trace_type_filter;
        self.compute.upload_trail_params(&self.queue);

        // 4) GPU step - only when GPU physics is enabled
        if ui.use_gpu_physics && (!ui.paused || ui.step_once) {
            self.compute.dispatch(&mut encoder, self.compute.particle_count);
            
            // Only run complex trail capture for Lines/Dots modes, not Simple mode
            if matches!(ui.trace_render_mode, crate::ui::TraceRenderMode::Lines | crate::ui::TraceRenderMode::Dots) {
                self.compute.dispatch_trail_capture(&mut encoder);
                self.compute.advance_trail_head();
                self.compute.upload_trail_params(&self.queue);
            }

            sim.step_count += 1;
            ui.step_once = false;
            
            // Debug output - print trail state once per second (roughly)
            if ui.debug_trails {
                use std::sync::atomic::{AtomicU32, Ordering};
                static DEBUG_COUNTER: AtomicU32 = AtomicU32::new(0);
                let counter = DEBUG_COUNTER.fetch_add(1, Ordering::Relaxed);
                if counter % 60 == 0 { // Assuming ~60 FPS
                    let trail_len = self.compute.trail_len;
                    let valid_segments = if self.compute.trail_valid_len > 1 { self.compute.trail_valid_len - 1 } else { 0 };
                    let trail_vertex_count = self.compute.particle_count * valid_segments * 2;
                    println!("TRAIL DEBUG: particles={}, trail_len={}, head={}, valid_len={}, enabled={}, vertex_count={}", 
                        self.compute.particle_count, trail_len, self.compute.trail_head, 
                        self.compute.trail_valid_len, self.compute.trails_enabled, trail_vertex_count);
                }
            }
        }

        // 5) GPU selection is handled in compute shader - no readback needed

        // 6) Draw world — always render directly to swapchain with hard clear
        let load_op = wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 });

        self.draw.render(
            &mut encoder,
            &view,
            &self.depth.view,
            &self.compute.particle_buf,
            self.compute.particle_count,
            ui,
            &screen_desc,
            &self.queue,
            self.compute.trail_valid_len,
            load_op,
            &view,  // always the swapchain view
            sim,
        );

        // 6) Draw egui on top
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, screen_desc.pixels_per_point);

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            self.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// GPU selection eliminates need for CPU readback
    #[allow(dead_code)]
    pub fn complete_readback(&mut self, _sim: &mut SimState) -> Result<(), wgpu::BufferAsyncError> {
        // No-op - GPU selection handles everything
        Ok(())
    }
}