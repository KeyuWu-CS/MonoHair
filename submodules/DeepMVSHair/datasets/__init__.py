from datasets.OccDataset import OccDataset
from datasets.OriDataset import OriDataset
from datasets.EvalDataset import EvalDataset
# from datasets.GrowDataset import GrowDataset


def get_dataset(name, args):
    if name == 'occ':
        return OccDataset(**args)
    elif name == 'ori':
        return OriDataset(**args)
    elif name == 'eval':
        return EvalDataset(**args)
    # elif name == 'grow':
    #     return GrowDataset(**args)
    else:
        raise RuntimeError("Dataset {} not available".format(name))