//! Headless regression checks. GPU tests are opt-in on machines with an adapter.
use crate::{
    renderer::{
        compute::{ComputePipeline, GpuParams, SelectionParams},
        draw::DrawPipeline,
    },
    sim::{GpuParticle, SimState},
};
fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("GPU or software adapter");
    eprintln!("GPU test adapter: {:?}", adapter.get_info());
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .unwrap()
}
fn config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: 1400,
        height: 900,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    }
}
fn step(gpu: &mut ComputePipeline, device: &wgpu::Device, queue: &wgpu::Queue, params: &GpuParams) {
    gpu.upload_params(queue, params);
    let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    gpu.dispatch_grid(&mut e, params.count, params.grid_res);
    gpu.dispatch(&mut e, params.count);
    gpu.swap_particle_buffers();
    gpu.dispatch_grid(&mut e, params.count, params.grid_res);
    gpu.dispatch_reactions(&mut e, params.count);
    gpu.swap_particle_buffers();
    queue.submit([e.finish()]);
}
#[test]
fn all_shaders_validate() {
    for (name, source) in [
        ("compute", include_str!("renderer/shaders/compute.wgsl")),
        ("vertex", include_str!("renderer/shaders/vertex.wgsl")),
        ("fragment", include_str!("renderer/shaders/fragment.wgsl")),
        ("selection", include_str!("renderer/shaders/selection.wgsl")),
        (
            "trail_line",
            include_str!("renderer/shaders/trail_line.wgsl"),
        ),
        (
            "trail_capture",
            include_str!("renderer/shaders/trail_capture.wgsl"),
        ),
    ] {
        let m = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    }
}
#[test]
#[ignore = "requires a Vulkan/Metal/DX12 adapter; run with --ignored"]
fn gpu_pipelines_and_cpu_equivalence() {
    let (device, queue) = device();
    let _draw = DrawPipeline::new(&device, &config());
    // Picking and hidden slicing use the same rotated coordinates as rendering.
    {
        let mut gpu = ComputePipeline::new(&device, 2);
        let mut a = crate::sim::Particle::new(glam::Vec3::new(0.2, 0.5, 0.5), 0);
        a.position[3] = 0.8;
        let mut b = a;
        b.position[0] = 0.7;
        b.position[3] = 0.1;
        gpu.upload_particles(
            &queue,
            &[
                GpuParticle::from_particle(&a),
                GpuParticle::from_particle(&b),
            ],
        );
        let mut nd = crate::multidim::ViewNd::default();
        nd.axes = [0, 1, 3];
        nd.angles[0][3] = 90.0;
        let mut s = SelectionParams::default();
        s.mode_flags = [1, 2, 0, 0];
        s.dimension_data[0] = 4;
        s.projection = nd.packed(4);
        s.projection_data[0] = 0.5;
        s.viewport = [100.0, 100.0, 0.0, 0.0];
        s.rect_min = [59.0, 24.0, 0.0, 0.0];
        s.rect_max = [61.0, 26.0, 0.0, 0.0];
        gpu.upload_selection_params(&queue, &s);
        let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gpu.dispatch_selection(&mut e, 1);
        queue.submit([e.finish()]);
        assert_eq!(gpu.readback_selection(&device, &queue, 2).unwrap(), vec![0]);
        s.slice_centers[3] = 0.8;
        s.slice_thickness[3] = 0.01;
        gpu.upload_selection_params(&queue, &s);
        let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        gpu.dispatch_selection(&mut e, 1);
        queue.submit([e.finish()]);
        assert!(gpu
            .readback_selection(&device, &queue, 2)
            .unwrap()
            .is_empty());
    }
    // Both particles enter reaction range only after integration.
    {
        let mut sim = SimState::new();
        sim.set_type_count(2);
        sim.particles.clear();
        sim.params.bounds = 20.0;
        sim.params.dt = 1.0;
        sim.params.force_scale = 0.0;
        sim.params.friction = 1.0;
        sim.params.max_speed = 10.0;
        sim.params.reactions_enabled = true;
        sim.params.mix_radius = 0.001;
        sim.params.reaction_probability = 1.0;
        let mut a = crate::sim::Particle::new(glam::Vec3::new(1.0, 1.0, 1.0), 0);
        a.velocity[0] = 0.1;
        let mut b = crate::sim::Particle::new(glam::Vec3::new(1.2, 1.0, 1.0), 1);
        b.velocity[0] = -0.1;
        sim.particles = vec![a, b];
        sim.reaction_table = vec![-1, 1, 0, -1];
        let mut gpu = ComputePipeline::new(&device, 2);
        gpu.upload_particles(
            &queue,
            &sim.particles
                .iter()
                .map(GpuParticle::from_particle)
                .collect::<Vec<_>>(),
        );
        gpu.upload_rules(&queue, &sim.force_matrix);
        gpu.upload_reactions(&queue, &sim.reaction_table);
        let mut params = GpuParams::from(&sim.params);
        params.count = 2;
        step(&mut gpu, &device, &queue, &params);
        let result = gpu.readback_particles(&device, &queue).unwrap();
        sim.step();
        assert_eq!([result[0].kind, result[1].kind], [1, 0]);
        assert_eq!([sim.particles[0].kind, sim.particles[1].kind], [1, 0]);
    }
    for dim in [3, 4, 8] {
        for grid in [false, true] {
            let mut sim = SimState::new();
            sim.params.bounds = 20.0;
            sim.params.dimension = dim;
            sim.params.r_max = 3.0;
            sim.params.mix_radius = 2.0;
            sim.params.gpu_grid = grid;
            sim.params.reactions_enabled = true;
            sim.params.reaction_probability = 1.0;
            sim.params.dt = 0.01;
            sim.particles.clear();
            sim.spawn_random(128);
            for (i, p) in sim.particles.iter_mut().take(12).enumerate() {
                for d in 0..dim {
                    p.position[d] = if i % 2 == 0 { 0.1 } else { 19.9 };
                }
            }
            for i in 0..sim.reaction_table.len() {
                sim.reaction_table[i] = (i % sim.params.type_count) as i32;
            }
            let mut gpu = ComputePipeline::new(&device, 128);
            gpu.upload_particles(
                &queue,
                &sim.particles
                    .iter()
                    .map(GpuParticle::from_particle)
                    .collect::<Vec<_>>(),
            );
            gpu.upload_rules(&queue, &sim.force_matrix);
            gpu.upload_reactions(&queue, &sim.reaction_table);
            gpu.upload_trace_lengths(&queue, &sim.trace_len_matrix);
            for _ in 0..3 {
                let mut params = GpuParams::from(&sim.params);
                params.count = 128;
                params.frame = sim.step_count as u32;
                step(&mut gpu, &device, &queue, &params);
                sim.step();
                let particles = gpu.readback_particles(&device, &queue).unwrap();
                for (i, (a, b)) in particles.iter().zip(&sim.particles).enumerate() {
                    assert_eq!(a.kind, b.kind, "kind {i}, {dim}D, grid={grid}");
                    for d in 0..dim {
                        let delta =
                            crate::session::wrapped_delta(a.position[d] - b.position[d], 20.0);
                        assert!(
                            delta.abs() < 1e-4,
                            "position {i}, axis {d}, {dim}D, grid={grid}: {delta}"
                        );
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "requires GPU/software Vulkan; optional PARTICLE_LIFE_PREVIEW path receives PPM"]
fn offscreen_multidimensional_ui() {
    let (device, queue) = device();
    let cfg = config();
    let size = wgpu::Extent3d {
        width: 1400,
        height: 900,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("UI preview"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: cfg.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());
    let mut sim = SimState::new();
    sim.set_dimension(5);
    let mut ui = crate::ui::UiState::new();
    ui.paused = true;
    ui.nd.open = true;
    ui.nd.axes = [0, 3, 4];
    ui.nd.angles[0][3] = 25.0;
    ui.session.open = true;
    let ctx = egui::Context::default();
    let input = || egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1400.0, 900.0),
        )),
        ..Default::default()
    };
    let first = ctx.run(input(), |ctx| crate::ui::draw_ui(ctx, &mut sim, &mut ui));
    let output = ctx.run(input(), |ctx| crate::ui::draw_ui(ctx, &mut sim, &mut ui));
    let mut gpu = ComputePipeline::new(&device, 512);
    gpu.upload_particles(
        &queue,
        &sim.particles
            .iter()
            .map(GpuParticle::from_particle)
            .collect::<Vec<_>>(),
    );
    let mut draw = DrawPipeline::new(&device, &cfg);
    draw.update_selection_bind_group(&device, &gpu.selection_buf);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [1400, 900],
        pixels_per_point: 1.0,
    };
    let mut e = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    draw.render(
        &mut e,
        &view,
        &depth_view,
        gpu.current_render_particle_buffer(),
        512,
        &mut ui,
        &screen,
        &queue,
        0,
        wgpu::LoadOp::Clear(wgpu::Color {
            r: 0.02,
            g: 0.02,
            b: 0.02,
            a: 1.0,
        }),
        &view,
        &sim,
    );
    let jobs = ctx.tessellate(output.shapes, 1.0);
    let mut egui = egui_wgpu::Renderer::new(&device, cfg.format, None, 1);
    for (id, delta) in first
        .textures_delta
        .set
        .iter()
        .chain(&output.textures_delta.set)
    {
        egui.update_texture(&device, &queue, *id, delta);
    }
    let buffers = egui.update_buffers(&device, &queue, &mut e, &jobs, &screen);
    queue.submit(buffers);
    {
        let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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
        egui.render(&mut pass, &jobs, &screen);
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 5632 * 900,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    e.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::ImageCopyBuffer {
            buffer: &staging,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(5632),
                rows_per_image: Some(900),
            },
        },
        size,
    );
    queue.submit([e.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let bytes = slice.get_mapped_range();
    assert!(bytes.iter().any(|&b| b > 100));
    if let Ok(path) = std::env::var("PARTICLE_LIFE_PREVIEW") {
        use std::io::Write;
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"P6\n1400 900\n255\n").unwrap();
        for row in bytes.chunks(5632) {
            for pixel in row[..5600].chunks(4) {
                f.write_all(&pixel[..3]).unwrap();
            }
        }
    }
}
