# New-style dissection experiment code.
import torch, argparse, os, shutil, inspect, json
from collections import defaultdict
from netdissect import pbar, nethook, renormalize, pidfile, zdataset
from netdissect import upsample, tally, imgviz, imgsave, bargraph
import setting
import netdissect
torch.backends.cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transforms
from netdissect import parallelfolder
import torch.nn as nn
from torchvision import ops

IM_SIZE = 512

def make_upfn(args, dataset, model, layername):
    '''Creates an upsampling function.'''
    convs, data_shape = None, None
    if args.model == 'alexnet':
        convs = [layer for name, layer in model.model.named_children()
                if name.startswith('conv') or name.startswith('pool')]
    elif args.model == 'progan':
        # Probe the data shape
        out = model(dataset[0][0][None,...].cuda())
        data_shape = model.retained_layer(layername).shape[2:]
        upfn = upsample.upsampler(
                (64, 64),
                data_shape=data_shape,
                image_size=out.shape[2:])
        return upfn
    else:
        # Probe the data shape
        _ = model(dataset[0][0][None,...].cuda())
        data_shape = model.retained_layer(layername).shape[2:]
        pbar.print('upsampling from data_shape', tuple(data_shape))
    upfn = upsample.upsampler(
            (IM_SIZE, IM_SIZE),
            data_shape=data_shape,
            source=dataset,
            convolutions=convs)
    return upfn

def instrumented_layername(args):
    '''Chooses the layer name to dissect.'''
    if args.layer is not None:
        if args.model == 'vgg16':
            return 'features.' + args.layer
        return args.layer
    # Default layers to probe
    if args.model == 'alexnet':
        return 'conv5'
    elif args.model == 'vgg16':
        return 'features.conv5_3'
    elif args.model == 'resnet152':
        return '7'
    elif args.model == 'progan':
        return 'layer4'

def graph_conceptcatlist(conceptcatlist, **kwargs):
    count = defaultdict(int)
    catcount = defaultdict(int)
    for c in conceptcatlist:
        count[c] += 1
    for c in count.keys():
        catcount[c[1]] += 1
    cats = ['object', 'part', 'material', 'texture', 'color']
    catorder = dict((c, i) for i, c in enumerate(cats))
    sorted_labels = sorted(count.keys(),
        key=lambda x: (catorder[x[1]], -count[x]))
    sorted_labels
    return bargraph.make_svg_bargraph(
        [label for label, cat in sorted_labels],
        [count[k] for k in sorted_labels],
        [(c, catcount[c]) for c in cats], **kwargs)

def save_conceptcat_graph(filename, conceptcatlist):
    svg = graph_conceptcatlist(conceptcatlist, barheight=80, file_header=True)
    with open(filename, 'w') as f:
        f.write(svg)


class FloatEncoder(json.JSONEncoder):
    def __init__(self, nan_str='"NaN"', **kwargs):
        super(FloatEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def iterencode(self, o, _one_shot=False):
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring
        def floatstr(o, allow_nan=self.allow_nan,
                _inf=json.encoder.INFINITY, _neginf=-json.encoder.INFINITY,
                nan_str=self.nan_str):
            if o != o:
                text = nan_str
            elif o == _inf:
                text = '"Infinity"'
            elif o == _neginf:
                text = '"-Infinity"'
            else:
                return repr(o)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))
            return text

        _iterencode = json.encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

def dump_json_file(target, data):
    with open(target, 'w') as f:
        json.dump(data, f, indent=1, cls=FloatEncoder)

def copy_static_file(source, target):
    sourcefile = os.path.join(
            os.path.dirname(inspect.getfile(netdissect)), source)
    shutil.copy(sourcefile, target)
