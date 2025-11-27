import torch
import numpy as np
from PIL import Image
#
import del_splat_dlpack.Points.torch
import del_splat_dlpack.SplatCircle.torch
import del_splat_dlpack.SplatPoint.torch

def save_tensor_image(t, path):
    # t: (H, W, C), float32, 0-1
    t = t.detach().cpu().clamp(0, 1)            # 念のため clamp
    t = (t * 255).byte()                        # uint8 に変換
    img = Image.fromarray(t.numpy())
    img.save(path)

def perspective_fov(f_mm, aspect, z_near, z_far):
    sensor_height_mm = 24.0
    fov_y_deg = 2.0 * np.degrees(np.arctan(sensor_height_mm / (2.0 * f_mm)))
    f = 1.0 / np.tan(np.radians(fov_y_deg) / 2.0)
    a = aspect
    nf = 1.0 / (z_near - z_far)

    return np.array([
        [f / a, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -(z_far + z_near) * nf, -(2 * z_far * z_near) * nf],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def modelview(t):
    return np.array([
        [1, 0, 0, -t[0]],
        [0, 1, 0, -t[1]],
        [0, 0, 1, -t[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)

    pnt2xyz, pnt2rgb = del_splat_dlpack.Points.torch.load_ply("../asset/juice_box.ply")
    pnt2xyz, pnt2rgb = pnt2xyz.to(device), pnt2rgb.to(device)

    aabb = torch.cat([pnt2xyz.min(dim=0).values, pnt2xyz.max(dim=0).values])

    img_shape = (1000, 600)
    mat_p = perspective_fov(50.0, float(img_shape[0])/float(img_shape[1]), 0.1, 2.0)
    center = ((aabb[0:3] + aabb[3:6])*0.5).cpu().numpy()
    cam_pos = np.array([center[0], center[1], center[2]+1.4])
    mat_mv = modelview(cam_pos)
    mat_mvp = mat_p @ mat_mv
    transform_world2ndc = torch.from_numpy(mat_mvp).to(device)

    pnt2pixxyndcz, pnt2pixrad = del_splat_dlpack.SplatCircle.torch.from_project_spheres(
        pnt2xyz,
        0.0015,
        transform_world2ndc,
        img_shape)

    pix2rgb = del_splat_dlpack.SplatPoint.torch.rasterize_pix(
        pnt2pixxyndcz.cpu(),
        pnt2rgb.cpu(),
        img_shape)

    save_tensor_image(pix2rgb, "../target/dlpack_juice_box_pix_sort.png")