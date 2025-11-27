import torch
from .. import util_torch
from .. import _CapsuleAsDLPack

def rasterize_pix(
        pnt2pixxyndcz,
        pnt2rgb,
        img_shape):
    #
    device = pnt2pixxyndcz.device
    num_pnt = pnt2pixxyndcz.shape[0]
    #
    assert pnt2rgb.shape == (num_pnt,3)
    assert pnt2rgb.device == device
    assert pnt2rgb.dtype == torch.float32
    #
    pix2rgb = torch.empty((img_shape[1], img_shape[0], 3), device=device, dtype=torch.float32)
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import SplatPoint

    SplatPoint.rasterize_pix(
        util_torch.to_dlpack_safe(pnt2pixxyndcz,stream_ptr),
        util_torch.to_dlpack_safe(pnt2rgb,stream_ptr),
        util_torch.to_dlpack_safe(pix2rgb,stream_ptr),
        stream_ptr)

    return pix2rgb


def rasterize_pix_zbuffer(
        pnt2pixxyndcz,
        pnt2rgb,
        img_shape):
    #
    device = pnt2pixxyndcz.device
    num_pnt = pnt2pixxyndcz.shape[0]
    #
    assert pnt2rgb.shape == (num_pnt,3)
    assert pnt2rgb.device == device
    assert pnt2rgb.dtype == torch.float32
    #
    pix2rgb = torch.empty((img_shape[1], img_shape[0], 3), device=device, dtype=torch.float32)
    pix2unitdepth = torch.empty((img_shape[1], img_shape[0]), device=device, dtype=torch.float32) # zbuffer
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import SplatPoint

    SplatPoint.rasterize_pix_zbuffer(
        util_torch.to_dlpack_safe(pnt2pixxyndcz,stream_ptr),
        util_torch.to_dlpack_safe(pnt2rgb,stream_ptr),
        util_torch.to_dlpack_safe(pix2rgb,stream_ptr),
        util_torch.to_dlpack_safe(pix2unitdepth,stream_ptr),
        stream_ptr)

    return pix2rgb