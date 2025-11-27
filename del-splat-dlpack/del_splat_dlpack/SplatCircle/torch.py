import torch
from .. import util_torch
from .. import _CapsuleAsDLPack


def from_project_spheres(
    pnt2xyz: torch.Tensor,
    radius: float,
    transform_world2ndc: torch.Tensor,
    img_shape):
    #
    device = pnt2xyz.device
    num_pnt = pnt2xyz.shape[0]
    #
    assert transform_world2ndc.device == device
    assert transform_world2ndc.shape == (4,4)
    assert transform_world2ndc.dtype == torch.float32
    #
    pnt2pixxyndcz = torch.empty((num_pnt,3), device=device, dtype=torch.float32)
    pnt2pixrad = torch.empty((num_pnt), device=device, dtype=torch.float32)
    #
    transform_world2ndc = transform_world2ndc.transpose(0,1).flatten()
    #
    stream_ptr = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_ptr = torch.cuda.current_stream(device).cuda_stream
    #
    from .. import SplatCircle

    SplatCircle.from_project_spheres(
       util_torch.to_dlpack_safe(pnt2xyz, stream_ptr),
       radius,
       util_torch.to_dlpack_safe(transform_world2ndc, stream_ptr),
       img_shape,
       util_torch.to_dlpack_safe(pnt2pixxyndcz, stream_ptr),
       util_torch.to_dlpack_safe(pnt2pixrad, stream_ptr),
       stream_ptr)

    return pnt2pixxyndcz, pnt2pixrad



