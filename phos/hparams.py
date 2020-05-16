"""
Derived idea from openai/jukebox/hparams.py.
"""
HPARAMS_REGISTRY = {}
DEFAULTS = {}


class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def setup_hparams(hparam_set_names, kwargs):
    """ Setup hyperparameters.
    Priority:
        1. user specified (i.e. kwargs)
        2. HPARAMS_REGISTRY
        3. DEFAULTS
    """
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(',')
    hparam_sets = [HPARAMS_REGISTRY[x.strip()] for x in hparam_set_names if x] + [kwargs]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f'{k} not in default args')
        H.update(**hps)
    H.update(**kwargs)
    return H


# def load_bipolar_reference_json(fn):
#     import json
#     with open(fn, 'r') as f:
#         content = json.load(f)
#     # assert
#     return content


proj_dev = Hyperparams(
    root            = './data',
    kind            = 'train',
    extra_dir       = './data_extra',
    force_update    = False,
    aug_shift       = True,
)
HPARAMS_REGISTRY['proj_dev'] = proj_dev

train = Hyperparams(
    # root            = './data',
    kind            = 'train',
    # extra_dir       = './data_extra',
    force_update    = False,
)
HPARAMS_REGISTRY['train'] = train


DEFAULTS['data'] = Hyperparams(
    root            = None,
    kind            = None,
    # sample_length   = 1,
    # sample_length   = 2,
    sample_length   = 4,
    force_update    = False,
    csv_file        = None,
    extra_dir       = None,
    montage         = None,
    aug_shift       = False,
    # desired_fs      = 250,
    desired_fs      = 256,
    # f_lowpass       = 2,
    # f_highpass      = 128,
)
