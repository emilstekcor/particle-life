pub mod compute;
pub mod draw;

use winit::window::Window;
use winit::event::Event;
use egui_wgpu::ScreenDescriptor;

use crate::sim::SimState;
use crate::ui::UiState;

/// Owns the wgpu device/queue/surface and all GPU pipelines.
pub struct Renderer {
    // Core wgpu handles
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    surface:     wgpu::Surface<'static>,
    surface_cfg: wgpu::SurfaceConfiguration,

    // Compute pipeline for particle physics
    pub compute: compute::ComputePipeline,

    // Render pipeline for drawing particles as points/billboards
    pub draw: draw::DrawPipeline,

    // egui integration
    egui_ctx:      egui::Context,
    egui_renderer: egui_wgpu::Renderer,
    egui_state:    egui_winit::State,
}

impl Renderer {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_cfg);

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
        let compute = compute::ComputePipeline::new(&device, 1024);
        let draw = draw::DrawPipeline::new(&device, &surface_cfg);

        Self {
            device,
            queue,
            surface,
            surface_cfg,
            compute,
            draw,
            egui_ctx,
            egui_renderer,
            egui_state,
        }
    }

    /// Forward window events to egui. Returns true if egui consumed the event.
    pub fn egui_handle_event(&mut self, window: &Window, event: &Event<()>) -> bool {
        if let winit::event::Event::WindowEvent { event, .. } = event {
            let response = self.egui_state.on_window_event(window, event);
            return response.consumed;
        }
        false
    }

    pub fn resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 { return; }
        self.surface_cfg.width = size.width;
        self.surface_cfg.height = size.height;
        self.surface.configure(&self.device, &self.surface_cfg);
    }

    /// Main render function called every frame.
    pub fn render(&mut self, window: &Window, sim: &mut SimState, ui: &mut UiState) {
        // 1. Run compute shader (physics step on GPU)
        self.compute.run(&self.device, &self.queue, sim);

        // 2. Get the next frame's texture to draw into
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

        // 3. Draw particles
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.surface_cfg.width, self.surface_cfg.height],
            pixels_per_point: window.scale_factor() as f32,
        };
        self.draw.render(&mut encoder, &view, &self.compute, sim, ui, &screen_desc, &self.queue);

        // 4. Draw egui UI on top
        let raw_input = self.egui_state.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            crate::ui::draw_ui(ctx, sim, ui);
        });

        self.egui_state.handle_platform_output(window, full_output.platform_output);

        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, screen_desc.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &paint_jobs, &screen_desc);

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // don't clear — draw on top of particles
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

        // 5. Submit and present
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}