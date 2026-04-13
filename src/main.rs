mod sim;
mod ui;
mod renderer;

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

    // Load saved creatures from book.json if it exists
    sim.book.load_from_file("book.json");

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);

        // Pass window events to egui first
        // If egui didn't consume the event, we handle it for the sim/camera
        let egui_consumed = renderer.egui_handle_event(&window, &event);

        match event {
            Event::WindowEvent { event: ref win_event, .. } => {
                match win_event {
                    WindowEvent::CloseRequested => target.exit(),

                    WindowEvent::Resized(size) => {
                        renderer.resize(*size);
                    }

                    WindowEvent::RedrawRequested => {
                        // 1. Step the simulation (unless paused)
                        if !ui.paused {
                            sim.step();
                        } else if ui.step_once {
                            sim.step();
                            ui.step_once = false;
                        }

                        // 2. Draw everything
                        renderer.render(&window, &mut sim, &mut ui);
                    }

                    _ => {}
                }
            }

            Event::AboutToWait => {
                // Request a redraw every frame
                window.request_redraw();
            }

            _ => {}
        }
    }).unwrap();
}
