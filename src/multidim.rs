//! Shared orthonormal projection for rendering, picking, slices and dragging.
use crate::{
    sim::{SimState, MAX_DIM},
    ui::UiState,
};
use serde::{Deserialize, Serialize};
pub const AXES: [&str; 8] = ["X", "Y", "Z", "W", "V", "U", "T", "S"];
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewNd {
    pub open: bool,
    pub axes: [usize; 3],
    pub angles: [[f32; 8]; 8],
    pub plane: [usize; 2],
    pub animate: bool,
    pub degrees_per_second: f32,
}
impl Default for ViewNd {
    fn default() -> Self {
        Self {
            open: false,
            axes: [0, 1, 2],
            angles: [[0.0; 8]; 8],
            plane: [0, 3],
            animate: false,
            degrees_per_second: 15.0,
        }
    }
}
impl ViewNd {
    pub fn normalize(&mut self, dim: usize) {
        let mut used = [false; 8];
        for axis in &mut self.axes {
            if *axis >= dim || used[*axis] {
                *axis = (0..dim).find(|&d| !used[d]).unwrap_or(0);
            }
            used[*axis] = true;
        }
        self.plane[0] = self.plane[0].min(dim - 1);
        self.plane[1] = self.plane[1].min(dim - 1);
        if self.plane[0] == self.plane[1] {
            self.plane[1] = (self.plane[0] + 1) % dim;
        }
    }
    pub fn order(&self, dim: usize) -> Vec<usize> {
        let mut v = self.clone();
        v.normalize(dim);
        let mut order = v.axes.to_vec();
        order.extend((0..8).filter(|d| !v.axes.contains(d)));
        order
    }
    pub fn basis(&self, dim: usize) -> [[f32; 8]; 8] {
        let mut b = [[0.0; 8]; 8];
        for d in 0..8 {
            b[d][d] = 1.0;
        }
        for a in 0..dim {
            for c in a + 1..dim {
                let (s, co) = self.angles[a][c].to_radians().sin_cos();
                for d in 0..8 {
                    let x = b[a][d];
                    let y = b[c][d];
                    b[a][d] = co * x - s * y;
                    b[c][d] = s * x + co * y;
                }
            }
        }
        let order = self.order(dim);
        std::array::from_fn(|r| b[order[r]])
    }
    pub fn packed(&self, dim: usize) -> [[f32; 4]; 16] {
        let b = self.basis(dim);
        std::array::from_fn(|i| std::array::from_fn(|j| b[i / 2][(i % 2) * 4 + j]))
    }
    pub fn unproject_delta(&self, delta: [f32; 3], dim: usize) -> [f32; 8] {
        let b = self.basis(dim);
        std::array::from_fn(|d| (0..3).map(|r| delta[r] * b[r][d]).sum())
    }
    pub fn project(&self, position: [f32; 8], dim: usize, bounds: f32) -> [f32; 8] {
        let b = self.basis(dim);
        let c = bounds * 0.5;
        std::array::from_fn(|r| c + (0..dim).map(|d| b[r][d] * (position[d] - c)).sum::<f32>())
    }
}
pub fn draw(ctx: &egui::Context, sim: &mut SimState, ui: &mut UiState) {
    ui.nd.normalize(sim.params.dimension);
    if ui.nd.animate {
        let a = ui.nd.plane[0].min(ui.nd.plane[1]);
        let b = ui.nd.plane[0].max(ui.nd.plane[1]);
        ui.nd.angles[a][b] = (ui.nd.angles[a][b]
            + ui.nd.degrees_per_second * ctx.input(|i| i.stable_dt).min(0.1)
            + 180.0)
            .rem_euclid(360.0)
            - 180.0;
        ctx.request_repaint();
    }
    if !ui.nd.open {
        return;
    }
    let mut open = true;
    egui::Window::new("Dimensions & projection").open(&mut open).default_pos([470.0,100.0]).default_width(370.0).default_height(610.0).vscroll(true).show(ctx,|e|{
  let dim=sim.params.dimension;let mut requested=dim;e.add(egui::Slider::new(&mut requested,3..=MAX_DIM).text("Physics dimensions"));if requested!=dim{ui.pending_dimension=Some(requested);}
  e.label("View controls change the projection, not the physics.");
  e.horizontal(|e|{for r in 0..3{egui::ComboBox::from_id_source(("display_axis",r)).selected_text(format!("{}: {}",["Horizontal","Vertical","Depth"][r],AXES[ui.nd.axes[r]])).show_ui(e,|e|{for d in 0..dim{if e.selectable_label(ui.nd.axes[r]==d,AXES[d]).clicked(){if let Some(other)=ui.nd.axes.iter().position(|&x|x==d){ui.nd.axes.swap(r,other);}else{ui.nd.axes[r]=d;}}}});}});
  e.horizontal(|e|{if e.button("XYZ").clicked(){ui.nd.axes=[0,1,2];}if dim>=4&&e.button("XYW").clicked(){ui.nd.axes=[0,1,3];}if e.button("Reset rotations").clicked(){ui.nd.angles=[[0.0;8];8];ui.nd.animate=false;}});
  e.separator();e.label("Rotate a plane");
  e.horizontal(|e|{for k in 0..2{egui::ComboBox::from_id_source(("rotation_plane",k)).selected_text(AXES[ui.nd.plane[k]]).show_ui(e,|e|{for d in 0..dim{if d!=ui.nd.plane[1-k]{e.selectable_value(&mut ui.nd.plane[k],d,AXES[d]);}}});}});
  let a=ui.nd.plane[0].min(ui.nd.plane[1]);let b=ui.nd.plane[0].max(ui.nd.plane[1]);e.add(egui::Slider::new(&mut ui.nd.angles[a][b],-180.0..=180.0).text("degrees"));e.checkbox(&mut ui.nd.animate,"Animate this plane");e.add(egui::Slider::new(&mut ui.nd.degrees_per_second,-90.0..=90.0).text("degrees / second"));
  e.separator();e.label("Hidden-axis slices (in rotated coordinates)");let order=ui.nd.order(dim);
  e.horizontal(|e|{if e.button("Show all").clicked(){ui.extra_slice_thickness=[f32::MAX;8];}if e.button("Center slices").clicked(){ui.extra_slice_centers=[sim.params.bounds*0.5;8];ui.extra_slice_thickness=[sim.params.bounds;8];}});
  for r in 3..dim{e.push_id(r,|e|{let mut enabled=ui.extra_slice_thickness[r]<1e20;if e.checkbox(&mut enabled,format!("Slice {} (rotated)",AXES[order[r]])).changed(){ui.extra_slice_thickness[r]=if enabled{sim.params.bounds*0.25}else{f32::MAX};}if enabled{let extent=sim.params.bounds*(dim as f32).sqrt();e.add(egui::Slider::new(&mut ui.extra_slice_centers[r],-extent..=extent).text("center"));e.add(egui::Slider::new(&mut ui.extra_slice_thickness[r],0.0001..=extent*2.0).logarithmic(true).text("width"));}});}
  if dim==3{e.label("Enable 4D–8D to expose hidden-axis slices.");}e.small("Depth selection follows the displayed depth axis. Dragging follows the rotated view plane.");
 });
    ui.nd.open = open;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reducing_dimension_keeps_axes_unique() {
        let mut v = ViewNd::default();
        v.axes = [7, 6, 0];
        v.normalize(3);
        let mut a = v.axes;
        a.sort();
        assert_eq!(a, [0, 1, 2]);
    }
    #[test]
    fn rotated_basis_is_orthonormal_and_drag_roundtrips() {
        let mut v = ViewNd::default();
        v.axes = [3, 5, 0];
        v.angles[0][3] = 37.0;
        v.angles[2][5] = -61.0;
        let b = v.basis(8);
        for i in 0..8 {
            for j in 0..8 {
                let dot: f32 = (0..8).map(|k| b[i][k] * b[j][k]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-5);
            }
        }
        let d = [0.2, -0.7, 0.9];
        let inv = v.unproject_delta(d, 8);
        for i in 0..3 {
            assert!(((0..8).map(|k| b[i][k] * inv[k]).sum::<f32>() - d[i]).abs() < 1e-5);
        }
    }
}
