//! Particle selection system
//!
//! Screen-space queries for rectangle, brush, and depth-slice selection modes.

use crate::sim::SimState;
use glam::{Mat4, Vec3, Vec4Swizzles};

/// Query particles within a screen-space rectangle
pub fn query_particles_in_rect(
    sim: &SimState,
    view_proj: Mat4,
    viewport: [u32; 2],
    rect: egui::Rect,
) -> Vec<usize> {
    sim.particles
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let sp = world_to_screen(view_proj, Vec3::from(p.position), viewport)?;
            rect.contains(sp).then_some(i)
        })
        .collect()
}

/// Query particles within a screen-space brush circle
pub fn query_particles_in_brush(
    sim: &SimState,
    view_proj: Mat4,
    viewport: [u32; 2],
    brush_center: egui::Pos2,
    brush_radius: f32,
) -> Vec<usize> {
    let r2 = brush_radius * brush_radius;
    sim.particles
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let sp = world_to_screen(view_proj, Vec3::from(p.position), viewport)?;
            let d2 = (sp.x - brush_center.x).powi(2) + (sp.y - brush_center.y).powi(2);
            (d2 <= r2).then_some(i)
        })
        .collect()
}

/// Query particles within a view-space depth slice
pub fn query_particles_in_slice(
    sim: &SimState,
    view_matrix: Mat4,
    slice_center: f32,
    slice_thickness: f32,
) -> Vec<usize> {
    let half = slice_thickness * 0.5;
    let min_d = slice_center - half;
    let max_d = slice_center + half;

    sim.particles
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let vp = view_matrix * Vec3::from(p.position).extend(1.0);
            (vp.z >= min_d && vp.z <= max_d).then_some(i)
        })
        .collect()
}

/// Find the nearest particle within a screen-space radius
pub fn nearest_particle_in_screen_radius(
    sim: &SimState,
    view_proj: Mat4,
    viewport: [u32; 2],
    mouse: egui::Pos2,
    radius: f32,
) -> Option<usize> {
    let r2 = radius * radius;
    let mut best: Option<(usize, f32)> = None;

    for (i, p) in sim.particles.iter().enumerate() {
        if let Some(sp) = world_to_screen(view_proj, Vec3::from(p.position), viewport) {
            let dx = sp.x - mouse.x;
            let dy = sp.y - mouse.y;
            let d2 = dx * dx + dy * dy;
            if d2 <= r2 {
                if best.map_or(true, |(_, bd)| d2 < bd) {
                    best = Some((i, d2));
                }
            }
        }
    }

    best.map(|(i, _)| i)
}

/// Get camera-plane right/up/forward axes from a view matrix
pub fn camera_plane_axes(view_matrix: glam::Mat4) -> (glam::Vec3, glam::Vec3, glam::Vec3) {
    let inv = view_matrix.inverse();
    let right = inv.x_axis.truncate().normalize();
    let up = inv.y_axis.truncate().normalize();
    let forward = -inv.z_axis.truncate().normalize();
    (right, up, forward)
}

/// Convert world position to egui screen coordinates
fn world_to_screen(view_proj: Mat4, pos: Vec3, viewport: [u32; 2]) -> Option<egui::Pos2> {
    let clip = view_proj * pos.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.xyz() / clip.w;
    Some(egui::pos2(
        (ndc.x * 0.5 + 0.5) * viewport[0] as f32,
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport[1] as f32,
    ))
}
