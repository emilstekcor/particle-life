//! Shared selection-space helpers.

/// Get camera-plane right/up/forward axes from a view matrix.
pub fn camera_plane_axes(view_matrix: glam::Mat4) -> (glam::Vec3, glam::Vec3, glam::Vec3) {
    let inverse = view_matrix.inverse();
    let right = inverse.x_axis.truncate().normalize();
    let up = inverse.y_axis.truncate().normalize();
    let forward = -inverse.z_axis.truncate().normalize();
    (right, up, forward)
}
