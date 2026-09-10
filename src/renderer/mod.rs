pub mod compute;
pub mod draw;

use egui_wgpu::ScreenDescriptor;
use winit::event::Event;
use winit::window::Window;

use crate::sim::SimState;
use crate::ui::UiState;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct DepthTexture {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl DepthTexture {
    fn create(device: &wgpu::Device, cfg: &wgpu::SurfaceConfiguration) -> Self {
        let size = wgpu::Extent3d {
            width: cfg.width.max(1),
            height: cfg.height.max(1),
            depth_or_array_layers: 1,
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
        Self {
            _texture: texture,
            view,
        }
    }
}

/// Owns the wgpu device/queue/surface and all GPU pipelines.
pub struct Renderer {
    // Core wgpu handles
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,
    depth: DepthTexture,

    // Compute pipeline for GPU physics
    pub compute: compute::ComputePipeline,

    // Render pipeline for drawing particles as points/billboards
    pub draw: draw::DrawPipeline,

    // egui integration
    egui_ctx: egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    last_gpu_physics: bool,
}

impl Renderer {
    /// Sync GPU selection flags into `ui.selected_indices`.
    /// Also refreshes `sim.particles` from the GPU first (when GPU physics is
    /// active) so that any CPU-side edit on the selection — delete, duplicate,
    /// move, assign type — operates on current positions instead of snapping
    /// the whole simulation back to a stale CPU copy.
    pub fn sync_selection(&mut self, sim: &mut SimState, ui: &mut UiState) {
        if !ui.selection_readback_needed {
            return;
        }
        ui.selection_readback_needed = false;

        if ui.use_gpu_physics {
            self.sync_particles_from_gpu(sim);
        }

        match self.compute.readback_selection(
            &self.device,
            &self.queue,
            self.compute.particle_count,
        ) {
            Ok(selected_indices) => {
                ui.selected_indices = selected_indices;
                log::debug!(
                    "GPU selection synced: {} particles selected",
                    ui.selected_indices.len()
                );
            }
            Err(e) => {
                log::warn!("Failed to read back selection: {:?}", e);
            }
        }
    }

    /// Refresh the CPU particle mirror from the current GPU buffer.
    /// Preserves CPU-only fields (`prefab_local_type`) and does NOT set
    /// `particles_dirty` — the GPU already has this exact data.
    pub fn sync_particles_from_gpu(&mut self, sim: &mut SimState) -> bool {
        if self.compute.particle_count == 0 {
            return true;
        }
        match self.compute.readback_particles(&self.device, &self.queue) {
            Ok(gpu_particles) => {
                for (p, g) in sim.particles.iter_mut().zip(gpu_particles.iter()) {
                    p.position = g.position;
                    p.velocity = g.velocity;
                    p.kind = g.kind; // GPU reactions can change kind
                    p.prefab_id = g.prefab_id;
                }
                true
            }
            Err(e) => {
                log::warn!("Failed to read back particles: {:?}", e);
                false
            }
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
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(window).unwrap())
        }
        .unwrap();

        // Adapter = handle to a physical GPU
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        log::info!("GPU: {:?}", adapter.get_info().name);

        // The complete 200k-particle × 20-sample trail history is 192 MB.
        // WGPU's portable default only grants 128 MiB per storage binding even
        // when the adapter supports more, so request the usable amount here.
        // ComputePipeline still derives a smaller safe trail cap when an
        // adapter genuinely cannot expose the complete history.
        let adapter_limits = adapter.limits();
        let desired_trail_binding_size =
            compute::trail_history_size(crate::sim::MAX_RENDER_PARTICLES, compute::MAX_TRAIL);
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .min(desired_trail_binding_size.min(u32::MAX as u64) as u32);

        // Device = logical GPU handle. Queue = command submission queue.
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                },
                None,
            )
            .await
            .unwrap();

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
        let egui_state =
            egui_winit::State::new(egui_ctx.clone(), egui_ctx.viewport_id(), window, None, None);

        // ── GPU pipelines ──────────────────────────────────────────────────────
        let compute = compute::ComputePipeline::new(&device, crate::sim::MAX_RENDER_PARTICLES);
        let mut draw = draw::DrawPipeline::new(&device, &surface_cfg);

        // Initialize bind groups with actual buffers
        draw.update_selection_bind_group(&device, &compute.selection_buf);
        draw.update_trail_bind_group(
            &device,
            &compute.trail_history_buf,
            &compute.trail_params_buf,
        );

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
            last_gpu_physics: true,
        }
    }

    /// Forward window events to egui. Returns EventResponse with consumed/repaint flags.
    pub fn egui_handle_event(
        &mut self,
        window: &Window,
        event: &Event<()>,
    ) -> egui_winit::EventResponse {
        if let winit::event::Event::WindowEvent { event, .. } = event {
            self.egui_state.on_window_event(window, event)
        } else {
            egui_winit::EventResponse::default()
        }
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame"),
            });

        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        // Every CPU edit transaction starts from current GPU state. Never
        // read back after editing: deletion can change index ownership.
        let raw_input = self.egui_state.take_egui_input(window);
        let interacting = !raw_input.events.is_empty();
        if self.last_gpu_physics
            && (interacting
                || ui.drag_mode == crate::ui::DragMode::MovingSelection
                || ui.session.needs_live_state()
                || ui.selection_readback_needed)
        {
            if !self.sync_particles_from_gpu(sim) {
                ui.flash("GPU synchronization failed; edit was not applied");
                return;
            }
        }
        if ui.selection_readback_needed {
            self.sync_selection(sim, ui);
        }
        let before_edit = if interacting {
            Some(crate::session::Snapshot::capture(sim, ui))
        } else {
            None
        };
        ui.trace_len_limit = self.compute.max_trail;
        ui.trace_len = ui.trace_len.clamp(1, ui.trace_len_limit);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            crate::ui::draw_ui(ctx, sim, ui);
        });

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);

        if let Some(snapshot) = before_edit {
            if (sim.particles_dirty
                || sim.params_dirty
                || sim.force_matrix_dirty
                || sim.reaction_table_dirty
                || sim.trace_len_matrix_dirty
                || ui.pending_dimension.is_some())
                && !ui.session.undo_requested
                && !ui.session.redo_requested
            {
                ui.session.push_undo(snapshot);
            }
        }
        crate::session::process(sim, ui);
        if self.last_gpu_physics != ui.use_gpu_physics {
            sim.trace_timers = vec![0; sim.particles.len()];
            sim.particles_dirty = true;
            sim.particles_replaced = true;
        }
        self.last_gpu_physics = ui.use_gpu_physics;

        // Dimension changes need a current CPU mirror before initializing new
        // axes; otherwise a stale mirror would rewind GPU-owned XYZ positions.
        if let Some(new_dimension) = ui.pending_dimension.take() {
            let old_dimension = sim.params.dimension;
            sim.set_dimension(new_dimension);
            ui.nd.normalize(new_dimension);
            if new_dimension > old_dimension {
                for d in old_dimension..new_dimension {
                    ui.extra_slice_centers[d] = sim.params.bounds * 0.5;
                    ui.extra_slice_thickness[d] = sim.params.bounds;
                }
            }
            ui.selected_indices.clear();
            ui.clear_selection_requested = true;
            self.compute.reset_trails();
            self.compute
                .clear_trail_history(&mut encoder, crate::sim::MAX_RENDER_PARTICLES);
            ui.flash(format!("Switched to {}D", sim.params.dimension));
        }

        // Consume a pending "clear selection" request set by the UI this frame
        // (new brush stroke, Clear button, or after a delete shifted indices).
        // queue.write_buffer is ordered before the encoder submit, so the clear
        // lands before this frame's selection dispatch.
        if std::mem::take(&mut ui.clear_selection_requested) && self.compute.particle_count > 0 {
            self.compute
                .clear_selection(&self.queue, self.compute.particle_count);
        }

        let steps = ui
            .playback
            .steps(ui.paused, std::mem::take(&mut ui.step_once), ui.strobe);
        if !ui.use_gpu_physics {
            if sim.particles_replaced || self.compute.particle_count as usize != sim.particles.len()
            {
                sim.trace_timers = vec![0; sim.particles.len()];
            }
            for _ in 0..steps {
                sim.step();
            }
            if steps > 0 {
                sim.particles_dirty = true;
            }
        }
        if sim.particles_dirty {
            let gpu_particles: Vec<crate::sim::GpuParticle> = sim
                .particles
                .iter()
                .map(crate::sim::GpuParticle::from_particle)
                .collect();

            let particle_count = gpu_particles.len() as u32;
            let old_particle_count = self.compute.particle_count;
            self.compute.upload_particles(&self.queue, &gpu_particles);

            // Reset trails if all particles are cleared
            if particle_count != old_particle_count || sim.particles_replaced {
                ui.selected_indices.clear();
                self.compute.clear_selection(&self.queue, particle_count);
                self.queue.write_buffer(
                    &self.compute.trace_timer_buf,
                    0,
                    bytemuck::cast_slice(&vec![0u32; particle_count as usize]),
                );
                self.compute.reset_trails();
                let temp_encoder = &mut encoder;
                self.compute
                    .clear_trail_history(temp_encoder, crate::sim::MAX_RENDER_PARTICLES);
            }

            let mut gpu_params = compute::GpuParams::from(&sim.params);
            gpu_params.count = particle_count;

            // Safety guard: disable GPU reactions above 10k particles to prevent stalls
            if ui.use_gpu_physics
                && sim.params.reactions_enabled
                && particle_count > 10_000
                && gpu_params.grid_res == 0
            {
                gpu_params.reactions_enabled = 0; // Force disable reactions in GPU params only
            }

            self.compute.upload_params(&self.queue, &gpu_params);

            sim.particles_dirty = false;
            sim.particles_replaced = false;
        }

        // Params are re-uploaded every frame rather than only when dirty: the
        // reaction gate needs a fresh frame counter each step, and a 64-byte
        // uniform write is free next to the physics dispatch.
        {
            let mut gpu_params = compute::GpuParams::from(&sim.params);
            gpu_params.count = self.compute.particle_count;
            gpu_params.frame = sim.step_count as u32;

            // Safety guard: disable GPU reactions above 10k particles to prevent stalls
            if ui.use_gpu_physics
                && sim.params.reactions_enabled
                && self.compute.particle_count > 10_000
                && gpu_params.grid_res == 0
            {
                gpu_params.reactions_enabled = 0; // Force disable reactions in GPU params only
            }

            self.compute.upload_params(&self.queue, &gpu_params);
            sim.params_dirty = false;
        }

        if sim.force_matrix_dirty {
            self.compute.upload_rules(&self.queue, &sim.force_matrix);
            sim.force_matrix_dirty = false;
        }

        if sim.reaction_table_dirty {
            self.compute
                .upload_reactions(&self.queue, &sim.reaction_table);
            sim.reaction_table_dirty = false;
            log::debug!("uploaded reaction table");
        }

        if sim.trace_len_matrix_dirty {
            self.compute
                .upload_trace_lengths(&self.queue, &sim.trace_len_matrix);
            sim.trace_len_matrix_dirty = false;
        }

        // Upload GPU selection parameters. Mouse coordinates from egui are in
        // points, so the shader gets the viewport in points too. view_proj is
        // last frame's camera (written by draw.rs), which is at most one frame
        // stale — imperceptible for interactive selection.
        let (_, view_matrix, view_proj) = self.draw.build_camera(ui, &screen_desc, sim);
        ui.view_matrix = view_matrix;
        ui.view_proj = view_proj;
        ui.gpu_selection_params.view_proj = view_proj.to_cols_array_2d();
        ui.gpu_selection_params.viewport = [
            self.surface_cfg.width as f32 / screen_desc.pixels_per_point,
            self.surface_cfg.height as f32 / screen_desc.pixels_per_point,
            0.0,
            0.0,
        ];
        ui.gpu_selection_params.mode_flags[1] = self.compute.particle_count;
        ui.gpu_selection_params.slice_centers = ui.extra_slice_centers;
        ui.gpu_selection_params.slice_thickness = ui.extra_slice_thickness;
        ui.gpu_selection_params.dimension_data[0] = sim.params.dimension as u32;
        ui.gpu_selection_params.projection = ui.nd.packed(sim.params.dimension);
        ui.gpu_selection_params.projection_data = [sim.params.bounds * 0.5, 0.0, 0.0, 0.0];
        self.compute
            .upload_selection_params(&self.queue, &ui.gpu_selection_params);

        // 3) Sync UI state into compute trail parameters
        // Clear trail history when enabling trails mid-sim to avoid garbage
        let trails_enabled =
            ui.trace_render_mode != crate::ui::TraceRenderMode::Off && !ui.trace_ui_edit_only;
        let trails_newly_enabled = trails_enabled && !self.compute.trails_enabled;
        if trails_newly_enabled {
            self.compute.reset_trails();
            self.compute
                .clear_trail_history(&mut encoder, crate::sim::MAX_RENDER_PARTICLES);
        }

        self.compute.trails_enabled = trails_enabled;
        let requested_trail_len = ui.trace_len.clamp(1, self.compute.max_trail);
        let trail_len_changed = requested_trail_len != self.compute.trail_len;
        self.compute.trail_len = requested_trail_len;
        if trail_len_changed {
            self.compute.reset_trails();
            self.compute
                .clear_trail_history(&mut encoder, crate::sim::MAX_RENDER_PARTICLES);
        }
        self.compute.trail_type_filter = ui.trace_type_filter;
        self.compute
            .upload_trail_params(&self.queue, ui.trace_trigger_only, ui.trace_fade_alpha);

        if ui.use_gpu_physics && steps > 0 {
            for _ in 0..steps {
                let mut params = compute::GpuParams::from(&sim.params);
                params.count = self.compute.particle_count;
                params.frame = sim.step_count as u32;
                if params.count > 10_000 && params.grid_res == 0 {
                    params.reactions_enabled = 0;
                }
                self.compute.upload_params(&self.queue, &params);
                self.compute
                    .dispatch_grid(&mut encoder, params.count, params.grid_res);
                self.compute.dispatch(&mut encoder, params.count);
                self.compute.swap_particle_buffers();
                self.compute
                    .dispatch_grid(&mut encoder, params.count, params.grid_res);
                self.compute.dispatch_reactions(&mut encoder, params.count);
                self.compute.swap_particle_buffers();
                // A separate submission orders each substep's uniform write.
                self.queue.submit([encoder.finish()]);
                encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Next step / draw"),
                    });
                sim.step_count += 1;
            }
        }
        if steps > 0 {
            if !ui.use_gpu_physics {
                self.queue.write_buffer(
                    &self.compute.trace_timer_buf,
                    0,
                    bytemuck::cast_slice(&sim.trace_timers),
                );
            }
            self.compute.update_trail_capture_bind_group(&self.device);
            if matches!(
                ui.trace_render_mode,
                crate::ui::TraceRenderMode::Lines | crate::ui::TraceRenderMode::Dots
            ) {
                self.compute.advance_trail_head();
                self.compute.upload_trail_params(
                    &self.queue,
                    ui.trace_trigger_only,
                    ui.trace_fade_alpha,
                );
                self.compute.dispatch_trail_capture(&mut encoder);
            }
        }

        if ui.debug_trails && steps > 0 && sim.step_count % 60 == 0 {
            log::debug!(
                "trail: particles={}, head={}, valid_len={}, enabled={}",
                self.compute.particle_count,
                self.compute.trail_head,
                self.compute.trail_valid_len,
                self.compute.trails_enabled
            );
        }

        // 5) Selection pass — dedicated compute pass against the current
        //    particle buffer. Runs regardless of pause state or physics
        //    backend; internally a no-op when no selection tool is active.
        self.compute
            .dispatch_selection(&mut encoder, ui.gpu_selection_params.mode_flags[0]);

        // 6) Draw world — always render directly to swapchain with hard clear
        let load_op = wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.02,
            g: 0.02,
            b: 0.02,
            a: 1.0,
        });

        self.draw.render(
            &mut encoder,
            &view,
            &self.depth.view,
            self.compute.current_render_particle_buffer(),
            self.compute.particle_count,
            ui,
            &screen_desc,
            &self.queue,
            self.compute.trail_valid_len,
            load_op,
            &view, // always the swapchain view
            sim,
        );

        // 6) Draw egui on top
        let paint_jobs = self
            .egui_ctx
            .tessellate(full_output.shapes, screen_desc.pixels_per_point);

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
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

            self.egui_renderer
                .render(&mut render_pass, &paint_jobs, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
