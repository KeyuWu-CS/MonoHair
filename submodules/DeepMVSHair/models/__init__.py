from models.OrientPifu import OrientPifu
from models.pifu_attn import SRF, Ori_attn, Occ_attn


def get_model(name, args):
    if name == "occ":
        return Occ_attn(**args)
    elif name == "ori":
        return Ori_attn(**args)
    else:
        raise RuntimeError("model \"{}\" not available".format(name))