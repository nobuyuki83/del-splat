import torch
from .. import util_torch
from .. import _CapsuleAsDLPack

def load_ply(
        path_file: str):
    from .. import Points

    cap_pnt2xyz, cap_pnt2rgb = Points.load_ply(path_file)
    pnt2xyz = torch.from_dlpack(_CapsuleAsDLPack(cap_pnt2xyz))
    pnt2rgb = torch.from_dlpack(_CapsuleAsDLPack(cap_pnt2rgb))
    return pnt2xyz, pnt2rgb

