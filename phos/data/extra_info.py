from pathlib import Path
import pandas as pd


__all__ = ['extract_calibration_info']


def extract_calibration_info(extra_dir, fn_summary='seizures_v34r.xlsx'):    
    summary_file = Path(extra_dir, fn_summary)
    if not summary_file.exists():
        raise FileNotFoundError('Summary file is not found at %s' % summary_file)

    from os import makedirs
    from time import strftime
    from shutil import move

    print('Extracting calibration information from "%s" ...' % summary_file)
    kinds = ['dev', 'train', 'eval']
    default_header = [
        'dev_test', 'Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'train', 'Unnamed: 5',
        'Unnamed: 6', 'Unnamed: 7', 'eval', 'Unnamed: 9', 'Unnamed: 10'
    ]
    new_header_dict = {
        'dev_test': 'dev_filename',
        'Unnamed: 1': 'dev_start',
        'Unnamed: 2': 'dev_stop',
        'train': 'train_filename',
        'Unnamed: 5': 'train_start',
        'Unnamed: 6': 'train_stop',
        'eval': 'eval_filename',
        'Unnamed: 9': 'eval_start',
        'Unnamed: 10': 'eval_stop',
    }
    dir_backup = Path(extra_dir, 'backup_%s' % strftime('%Y%m%d%H%M%S'))

    df = pd.read_excel(summary_file, sheet_name='Calibration')
    if set(df.columns) != set(default_header):
        raise ValueError('Cannot process file. Columns of sheet might be modified.')

    # Drop empty columns and unused rows (nested header)
    df.drop(columns=['Unnamed: 3', 'Unnamed: 7'], index=[0, 1], inplace=True)
    # Rename columns
    df.rename(columns=new_header_dict, inplace=True)
    # Reset index
    df.reset_index(drop=True, inplace=True)

    sub_header = ['filename', 'start', 'stop']
    for kind in kinds:
        sel_cols = ['%s_%s' % (kind, v) for v in sub_header]
        data = df[sel_cols].dropna()
        data[sel_cols[0]] = data[sel_cols[0]].apply(lambda x: Path(x).with_suffix('').name)
        fn = 'calibration_%s.csv' % kind
        calb_file = Path(extra_dir, fn)
        print('Saving file to: "%s"' % calb_file)
        if calb_file.exists():
            print('Found existing file, it will be moved to %s' % dir_backup)
            if not dir_backup.exists():
                makedirs(dir_backup)
            move(calb_file, Path(dir_backup, fn))
        data.to_csv(calb_file, header=sub_header, index=False)
