def load_ply(path_file: str):
    from ..del_splat_dlpack import points_load_ply

    return points_load_ply(path_file)
