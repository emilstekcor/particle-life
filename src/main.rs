mod sim;
mod ui;
mod renderer;
mod selection;
mod crash_profile;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use sim::SimState;
use renderer::Renderer;
use ui::UiState;

fn main() {
    env_logger::init();

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

    event_loop.run(move |event, target| {
        // Pass window events to egui first
        let egui_resp = renderer.egui_handle_event(&window, &event);
        if egui_resp.repaint { 
            window.request_redraw(); 
        }

        match event {
            Event::WindowEvent { event: ref win_event, .. } => {
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
                                // CPU physics
                                sim.step();
                                sim.particles_dirty = true;
                            }
                        }

                        // Sync GPU selection with CPU selected_indices only when needed and not during active selection
                        if ui.selection_readback_needed && !ui.is_selecting {
                            renderer.sync_selection(&mut ui);
                        }
                        
                        renderer.render(&window, &mut sim, &mut ui);
                        
                        // Save crash profile after each frame for debugging
                        crash_profile::save_crash_profile(&sim, &ui);
                        
                        // Flush book saving if dirty (deferred I/O to avoid blocking UI)
                        sim.book.flush_if_dirty();
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
    }).unwrap();
}
