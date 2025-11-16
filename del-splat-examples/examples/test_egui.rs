use eframe::egui::Color32;
use eframe::{
    egui,
    egui::{ColorImage, TextureHandle},
};

// ----------

pub struct MyApp {
    texture: Option<TextureHandle>,
    point2xyz: Vec<f32>,
    point2rgb: Vec<f32>,
}

impl MyApp {
    fn initialize(&mut self) -> eframe::Result<()> {
        // let path = "/Users/nobuyuki/project/juice_box1.ply";
        let file_path = "asset/juice_box.ply";
        (self.point2xyz, self.point2rgb) = del_splat_cpu::io_ply::read_xyzrgb(file_path).unwrap();
        Ok(())
    }

    fn render(&self, img_shape: (f32, f32)) -> anyhow::Result<ColorImage> {
        let aabb3 = del_msh_cpu::vtx2xyz::aabb3(&self.point2xyz, 0f32);
        use num_traits::cast::AsPrimitive;
        let aabb3: [f32; 6] = aabb3.map(|v| v.as_());
        let transform_world2ndc = {
            let cam_proj = del_geo_core::mat4_col_major::camera_perspective_blender(
                img_shape.0 / img_shape.1,
                50f32,
                0.1,
                3.0,
                true,
            );
            let cam_modelview = del_geo_core::mat4_col_major::camera_external_blender(
                &[
                    (aabb3[0] + aabb3[3]) * 0.5f32,
                    (aabb3[1] + aabb3[4]) * 0.5f32,
                    (aabb3[2] + aabb3[5]) * 0.5f32 + 1.4f32,
                ],
                0f32,
                0f32,
                0f32,
            );
            del_geo_core::mat4_col_major::mult_mat_col_major(&cam_proj, &cam_modelview)
        };
        let transform_ndc2pix = del_geo_core::mat2x3_col_major::transform_ndc2pix((
            img_shape.0 as usize,
            img_shape.1 as usize,
        ));
        let radius = 0.0015;

        let num_point = self.point2xyz.len() / 3;
        let mut pnt2pixxydepth: Vec<f32> = vec![0f32; num_point * 3];
        let mut pnt2pixrad: Vec<f32> = vec![0f32; num_point];
        for i_elem in 0..num_point {
            let pos_world0 = arrayref::array_ref![self.point2xyz, i_elem * 3, 3];
            let pos_world0: [f32; 3] = pos_world0.map(|v| v.as_());
            let ndc0 = del_geo_core::mat4_col_major::transform_homogeneous(
                &transform_world2ndc,
                &pos_world0,
            )
            .unwrap();
            let pos_pix = del_geo_core::mat2x3_col_major::mult_vec3(
                &transform_ndc2pix,
                &[ndc0[0], ndc0[1], 1f32],
            );
            let rad_pix = {
                let dqdp = del_geo_core::mat4_col_major::jacobian_transform(
                    &transform_world2ndc,
                    &pos_world0,
                );
                let dqdp = del_geo_core::mat3_col_major::try_inverse(&dqdp).unwrap();
                let dx = [dqdp[0], dqdp[1], dqdp[2]];
                let dy = [dqdp[3], dqdp[4], dqdp[5]];
                let rad_pix_x =
                    (1.0 / del_geo_core::vec3::norm(&dx)) * 0.5 * img_shape.0 as f32 * radius;
                let rad_pxi_y =
                    (1.0 / del_geo_core::vec3::norm(&dy)) * 0.5 * img_shape.1 as f32 * radius;
                0.5 * (rad_pix_x + rad_pxi_y)
            };
            pnt2pixxydepth[i_elem * 3 + 0] = pos_pix[0];
            pnt2pixxydepth[i_elem * 3 + 1] = pos_pix[1];
            pnt2pixxydepth[i_elem * 3 + 2] = ndc0[2];
            pnt2pixrad[i_elem] = rad_pix;
        }

        let img_width = img_shape.0 as usize;
        let img_height = img_shape.1 as usize;
        let num_pixel = img_width * img_height;
        let mut img_data = vec![[0f32; 3]; num_pixel];
        del_splat_cpu::splat_point2::draw_pix_sort_z_(
            &pnt2pixxydepth,
            &self.point2rgb,
            img_width,
            &mut img_data,
        )?;
        let mut pixels = vec![egui::Color32::BLACK; num_pixel];
        for i_pix in 0..img_data.len() {
            let rgb = img_data[i_pix];
            let r = (rgb[0] * 255.0) as u8;
            let g = (rgb[1] * 255.0) as u8;
            let b = (rgb[2] * 255.0) as u8;
            let c = Color32::from_rgb(r, g, b);
            pixels[i_pix] = c;
        }
        let c = ColorImage {
            size: [img_width, img_height],
            pixels,
        };
        Ok(c)
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let img_shape = ui.available_size();
            self.render((img_shape.x, img_shape.y)).unwrap();
            // テクスチャ作成（最初の1回だけ）
            if self.texture.is_none() {
                let img_width = img_shape.x as usize;
                let img_height = img_shape.y as usize;
                let num_pixel = img_width * img_height;
                let pixels = vec![egui::Color32::BLACK; num_pixel];
                let c = ColorImage {
                    size: [img_width, img_height],
                    pixels,
                };
                self.texture =
                    Some(ctx.load_texture("dynamic", c, egui::TextureOptions::default()));
            }
            let new_image = self.render((img_shape.x, img_shape.y)).unwrap();
            // 毎フレーム更新
            if let Some(tex) = &mut self.texture {
                tex.set(new_image, egui::TextureOptions::NEAREST);
                ui.image(&tex.clone());
            }
        });
        ctx.request_repaint(); // 毎フレーム再描画
    }
}

fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions::default();
    let mut app = Box::new(MyApp {
        texture: None,
        point2xyz: vec![],
        point2rgb: vec![],
    });
    app.as_mut().initialize().unwrap();
    eframe::run_native(
        "Texture Update Demo",
        native_options,
        Box::new(|_cc| Ok(app)),
    )
}
