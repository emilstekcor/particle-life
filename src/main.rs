mod audio;
mod crash_profile;
mod renderer;
mod selection;
mod sim;
mod ui;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use renderer::Renderer;
use sim::SimState;
use ui::UiState;

fn main() {
    env_logger::init();
    
    // Debug: Print struct sizes to verify alignment
    println!("=== STRUCT SIZE VERIFICATION ===");
    println!("SelectionParams: {} bytes", std::mem::size_of::<renderer::compute::SelectionParams>());
    println!("TrailPoint: {} bytes", std::mem::size_of::<renderer::compute::TrailPoint>());
    println!("Particle: {} bytes", std::mem::size_of::<sim::Particle>());
    println!("GpuParticle: {} bytes", std::mem::size_of::<sim::GpuParticle>());
    println!("GpuParams: {} bytes", std::mem::size_of::<renderer::compute::GpuParams>());
    println!("TrailParams: {} bytes", std::mem::size_of::<renderer::compute::TrailParams>());
    println!("================================");

    // Set up panic hook to show crash profile location on panic
    std::panic::set_hook(Box::new(|info| {
        eprintln!("panic: {info}");
        eprintln!(
            "last crash profile: {:?}",
            crate::crash_profile::crash_profile_path()
        );
    }));

    // Build the OS window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Particle Life 3D")
        .with_inner_size(winit::dpi::LogicalSize::new(1400u32, 900u32))
        .build(&event_loop)
        .unwrap();

    // Initialize wgpu + egui + sim — all blocking via pollster
    let mut renderer = pollster::block_on(Renderer::new(&window));
    let mut sim = SimState::new();
    let mut ui = UiState::new();

    // Load saved creatures from user data directory
    let book_path = dirs::data_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap())
        .join("particle_life")
        .join("book.json");

    // Create directory if it doesn't exist
    if let Some(parent) = book_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    sim.book.load_from_file(&book_path.to_string_lossy());

    event_loop
        .run(move |event, target| {
            // Pass window events to egui first
            let egui_resp = renderer.egui_handle_event(&window, &event);
            if egui_resp.repaint {
                window.request_redraw();
            }

            match event {
                Event::WindowEvent {
                    event: ref win_event,
                    ..
                } => {
                    match win_event {
                        WindowEvent::CloseRequested => target.exit(),

                        WindowEvent::Resized(size) => {
                            renderer.resize(*size);
                        }

                        WindowEvent::RedrawRequested => {
                            // Physics stepping control
                            let should_step = if ui.paused {
                                std::mem::take(&mut ui.step_once)
                            } else {
                                true
                            };

                            if should_step {
                                if ui.use_gpu_physics {
                                    // GPU physics is handled in the renderer
                                    // step_count will be incremented there
                                } else {
                                    // CPU physics. Strobe runs two sim steps per
                                    // rendered frame so period-2 oscillating
                                    // objects appear frozen; a manual Step while
                                    // paused advances one step (flips the phase).
                                    let steps = if ui.strobe && !ui.paused { 2 } else { 1 };
                                    for _ in 0..steps {
                                        sim.step();
                                    }
                                    sim.particles_dirty = true;
                                }
                            }

                            // Selection readback + CPU/GPU particle sync are
                            // handled inside renderer.render (once per
                            // selection gesture, not per frame).
                            renderer.render(&window, &mut sim, &mut ui);

                            // Flush book saving if dirty (deferred I/O to avoid blocking UI)
                            sim.book.flush_if_dirty();

                            // ── Profiles ────────────────────────────────────
                            if std::mem::take(&mut ui.save_profile_now) {
                                crash_profile::save_crash_profile(&sim, &ui);
                                ui.flash("Profile saved");
                                log::info!(
                                    "profile saved to {:?}",
                                    crash_profile::crash_profile_path()
                                );
                            }
                            if ui.auto_save_profiles
                                && should_step
                                && sim.step_count > 0
                                && sim.step_count % ui.auto_save_interval as u64 == 0
                            {
                                crash_profile::save_crash_profile(&sim, &ui);
                            }
                        }

                        _ => {}
                    }
                }

                Event::AboutToWait => {
                    let animating = !ui.paused || ui.step_once || ui.camera_mode;
                    if animating {
                        target.set_control_flow(ControlFlow::Poll);
                        window.request_redraw();
                    } else {
                        target.set_control_flow(ControlFlow::Wait);
                    }
                }

                _ => {}
            }
        })
        .unwrap();
}



