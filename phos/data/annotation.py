import re
import numpy as np


__all__ = ['AnnotationProcessor']


def _prepare_re_pattern():
    sep = r'\,\s'
    p_lv = r'(\d+)'
    p_t = r'(\d+\.\d+)'
    p_ch = r'(\d+)'
    p_probs = r'\[([\d\,\s\.]+)\]'
    return sep.join([p_lv, p_lv, p_t, p_t, p_ch, p_probs])


class AnnotationProcessor(object):
    INPUT_FMT = np.dtype([
        ('lv', np.uint8), ('sublv', np.uint8), ('ts', '<f2'), ('te', '<f2'),
        ('ch', np.uint8), ('probs', object)
    ])
    OUTPUT_FMT = np.dtype([
        ('ch', np.uint8), ('anno', np.uint8), ('ts', '<f2'), ('te', '<f2'),
        ('prob', '<f2')
    ])
    PATTERN = _prepare_re_pattern()
    def __init__(self, pattern=None, input_fmt=None, output_fmt=None):
        self.pattern = pattern if pattern else self.PATTERN
        self.regex = re.compile(self.pattern)
        self.input_fmt = input_fmt if input_fmt else self.INPUT_FMT
        self.output_fmt = output_fmt if output_fmt else self.OUTPUT_FMT

    def convert(self, fn):
        with open(fn, 'r') as f:
            content = f.read()
            seq = self.regex.findall(content)
            ary = np.array(seq, dtype=self.input_fmt)

        # convert strings of probs to array
        for i in range(len(ary)):
            ary[i]['probs'] = np.fromstring(ary[i]['probs'], sep=',', dtype='f2')

        # convert object arrays to normal ndarray
        probs = np.stack(ary['probs'])
        max_indices = probs.argmax(axis=1)

        outputs = np.empty(ary.size, dtype=self.output_fmt)
        outputs[['ch', 'ts', 'te']] = ary[['ch', 'ts', 'te']]
        outputs['anno'] = max_indices
        outputs['prob'] = probs[np.arange(len(max_indices)), max_indices]
        return outputs

    def add_annotation(self, converted, anno, to_channels=None, sort_by=None,
        ignore_duplicate=False):
        """ Add an annotation to converted annotation array.

        Parameters
        ----------
        converted : np.ndarray
            Data converted by this processor.
        anno : dict
            Annotation to be added. (with all fields listed in `output_fmt`
            but excluding channel)
        to_channels : int, list of int, or None. optional.
            Channels of annotation to be added. Use `None` to add annotations
            to all available channels.
        sort_by : str or list of str
            Field name(s) for ordering.
        ignore_duplicate : bool
            Set this flag to False to raise an error when there are existing
            annotations.
        """
        if converted.dtype != self.output_fmt:
            raise TypeError(
                '`dtype` of given coverted data should be {}.'.format(self.output_fmt)
            )

        ch_set = set(converted['ch'])
        if to_channels is None:
            to_channels = list(ch_set)
        elif isinstance(to_channels, list):
            if not set(to_channels).issubset(ch_set):
                raise ValueError('Not all given channels exist.')
            to_channels = list(set(to_channels))
        elif isinstance(to_channels, int):
            to_channels = list(to_channels)
        else:
            raise ValueError('Invalid type of `to_channels`.')

        new_anno = np.empty(len(to_channels), dtype=self.output_fmt)
        for k, v in anno.items():
            new_anno[k] = v
        new_anno['ch'] = to_channels

        if not ignore_duplicate:
            # Check whether there are duplicate annotations
            check_set = converted[np.where(converted['anno'] == anno['anno'])]
            for v in new_anno:
                if v in check_set:
                    raise RuntimeError('Found possible duplicate: %s' % v)

        extended = np.concatenate((converted, new_anno))
        if sort_by:
            extended = extended[np.argsort(extended, order=sort_by)]
        return extended
