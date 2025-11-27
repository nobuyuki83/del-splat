

def from_project_spheres(
    pnt2xyz,
    radius: float,
    transform_world2ndc,
    img_shape,
    pnt2pixxyndcz,
    pnt2pixrad,
    stream_ptr=0):
    #
    from ..del_splat_dlpack import circle_splat_from_project_spheres

    return circle_splat_from_project_spheres(pnt2xyz, radius, transform_world2ndc, img_shape, pnt2pixxyndcz, pnt2pixrad, stream_ptr)