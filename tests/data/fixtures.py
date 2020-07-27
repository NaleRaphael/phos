from pathlib import Path


THIS_FILE = Path(__file__)
data_dir = Path(THIS_FILE.parent, './data/')
case_dict = {
    'normal': {
        'montage_config': 'tcp_ar',
        'fn_edf': data_dir.joinpath('normal_tcp_ar.edf'),
        'fn_lbl': data_dir.joinpath('normal_tcp_ar.lbl'),
    }
}
