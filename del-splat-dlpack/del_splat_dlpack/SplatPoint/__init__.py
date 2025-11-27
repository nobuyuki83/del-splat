def rasterize_pix(
    pnt2pixxyndcz,
    pnt2rgb,
    pix2rgb,
    stream_ptr=0):
    #
    from ..del_splat_dlpack import point_splat_rasterize_pix

    point_splat_rasterize_pix(pnt2pixxyndcz, pnt2rgb, pix2rgb, stream_ptr)


def rasterize_pix_zbuffer(
    pnt2pixxyndcz,
    pnt2rgb,
    pix2rgb,
    pix2unitdepth,
    stream_ptr=0):
    #
    from ..del_splat_dlpack import point_splat_rasterize_pix_zbuffer

    point_splat_rasterize_pix_zbuffer(pnt2pixxyndcz, pnt2rgb, pix2rgb, pix2unitdepth, stream_ptr)