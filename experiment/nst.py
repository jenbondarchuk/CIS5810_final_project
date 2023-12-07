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
DATASET = "dancer"
MODEL = "flat"
quantile=0.01
miniou = 0.04
thumbsize=500
seg = "netpqc" # ['net', 'netp', 'netq', 'netpq', 'netpqc', 'netpqxc']

# Layers we added content/style loss to for reference
# unflattened:
# content_layers_default = ['layer_8', 'layer_14']
# style_layers_default = ['layer_1', 'layer_2', 'layer_4', 'layer_8', 'layer_14']

# flattened:
# content_layers_default = ['conv_8_2', 'conv_14_2']
# style_layers_default = ['conv_1_1', 'conv_2_2', 'conv_4_2', 'conv_8_2', 'conv_14_2']
layername = "conv_8_2"

# run on animation (content) image from Ishita's notebook, with Impressionist style image
# also run on picasso & dancer

# input content image and style image and input

def load_nst_data(path_to_data=f"/home/jen/Documents/git/dissect/experiment/datasets/{DATASET}"):
    """Load images."""
    imsize = IM_SIZE if torch.cuda.is_available() else 128  # use small size if no GPU

    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])

    # Use their loader, pass our transform here
    return parallelfolder.ParallelImageFolders([path_to_data],
                classification=True,
                shuffle=True,
                transform=loader)

def load_nst_model(path_to_model="/home/jen/Downloads/mobilenetv2_finetuned.pt"):
    """Load unflattened model, wrap with instrumented model so we can do network dissection."""
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path_to_model))
    model = nethook.InstrumentedModel(model).cuda().eval()
    return model

def get_name_type(layer):
    if isinstance(layer, nn.Conv2d):
        name = f'conv'
    elif isinstance(layer, nn.ReLU6):
        name = f'relu'
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = f'pool'
    elif isinstance(layer, nn.BatchNorm2d):
        name = f'bn'
    elif isinstance(layer, ops.Conv2dNormActivation):
        name = "block"
    else:
        raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
    return name

def load_flattened_nst_model(path_to_model="/home/jen/Downloads/mobilenetv2_finetuned.pt"):
    cnn = models.mobilenet_v2(pretrained=True)
    cnn.classifier[1] = torch.nn.Linear(cnn.classifier[1].in_features, 2)
    cnn.load_state_dict(torch.load(path_to_model))
    cnn = cnn.features[:15].eval()

    model_flat = nn.Sequential()
    for block_num, block in enumerate(cnn.children()):
        layer_num = 0
        for layers in block.children():
            if isinstance(layers, nn.Sequential):
                for layer in layers.children():
                    name_type = get_name_type(layer)
                    if name_type == 'block':
                        for sublayer in layer.children():
                            sub_name_type = get_name_type(sublayer)
                            model_flat.add_module(f'{sub_name_type}_{block_num}_{layer_num}', sublayer)
                    else:
                        model_flat.add_module(f'{name_type}_{block_num}_{layer_num}', layer)
                    layer_num += 1
            else:
                name_type = get_name_type(layers)
                model_flat.add_module(f'{name_type}_{block_num}_{layer_num}', layers)
                layer_num += 1

    model_flat = nethook.InstrumentedModel(model_flat).cuda().eval()
    return model_flat

def main():
    resdir = 'results/%s-%s-%s-%s' % (MODEL, DATASET, seg, layername)
    resfile = pidfile.exclusive_dirfn(resdir)

    # load model
    model = load_flattened_nst_model()

    # use feature.<layer_numer> for unflattened
    # use <op_name>_<layer_num>_<index_in_layer> for flattened
    
    model.retain_layer(layername)
    print("Layer of interest: ", layername)

    # load dataset
    dataset = load_nst_data() 

    # set up remaining functions/arguments for network dissection
    upfn = make_upfn(dataset, model, layername)
    sample_size = len(dataset)
    percent_level = 1.0 - quantile
    iou_threshold = miniou
    image_row_width = 3
    torch.set_grad_enabled(False)

    # Tally rq.np (representation quantile, unconditional).
    pbar.descnext('rq') # just changing progress bar/status message
    def compute_samples(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])
    rq = tally.tally_quantile(compute_samples, dataset,
                              sample_size=sample_size,
                              r=8192,
                              num_workers=100,
                              pin_memory=True,
                              cachefile=resfile('rq.npz'))

    # Create visualizations - first we need to know the topk
    pbar.descnext('topk') # top k values
    def compute_image_max(batch, *args):
        data_batch = batch.cuda()
        _ = model(data_batch)
        acts = model.retained_layer(layername)
        acts = acts.view(acts.shape[0], acts.shape[1], -1)
        acts = acts.max(2)[0]
        return acts
    topk = tally.tally_topk(compute_image_max, dataset, sample_size=sample_size,
            batch_size=50, num_workers=30, pin_memory=True,
            cachefile=resfile('topk.npz'))

    # Visualize top-activating patches of top-activating images.
    pbar.descnext('unit_images')
    image_size, image_source = None, None
    image_source = dataset
    iv = imgviz.ImageVisualizer((thumbsize, thumbsize),
        image_size=image_size,
        source=dataset,
        quantiles=rq,
        level=rq.quantiles(percent_level))
    def compute_acts(data_batch, *ignored_class):
        data_batch = data_batch.cuda()
        acts_batch = model.retained_layer(layername)
        return (acts_batch, data_batch)
    unit_images = iv.masked_images_for_topk(
            compute_acts, dataset, topk,
            k=image_row_width, num_workers=30, pin_memory=True,
            cachefile=resfile('top%dimages.npz' % image_row_width))
    pbar.descnext('saving images')
    imgsave.save_image_set(unit_images, resfile('image/unit%d.jpg'),
            sourcefile=resfile('top%dimages.npz' % image_row_width))

    # Compute IoU agreement between segmentation labels and every unit
    # Grab the 99th percentile, and tally conditional means at that level.
    level_at_99 = rq.quantiles(percent_level).cuda()[None,:,None,None]

    segmodel, seglabels, segcatlabels = setting.load_segmenter(seg)
    renorm = renormalize.renormalizer(dataset, target='zc')
    def compute_conditional_indicator(batch, *args):
        data_batch = batch.cuda()
        image_batch = renorm(data_batch)
        seg = segmodel.segment_batch(image_batch, downsample=1)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        iacts = (hacts > level_at_99).float() # indicator
        return tally.conditional_samples(iacts, seg)

    # Will need to customize this if we have time - will do once
    # we finalize a dataset
    pbar.descnext('condi99')
    condi99 = tally.tally_conditional_mean(compute_conditional_indicator,
            dataset, sample_size=sample_size,
            num_workers=3, pin_memory=True,
            cachefile=resfile('condi99.npz'))

    # Now summarize the iou stats and graph the units
    iou_99 = tally.iou_from_conditional_indicator_mean(condi99)
    unit_label_99 = [
            (concept.item(), seglabels[concept],
                segcatlabels[concept], bestiou.item())
            for (bestiou, concept) in zip(*iou_99.max(0))]
    labelcat_list = [labelcat
            for concept, label, labelcat, iou in unit_label_99
            if iou > iou_threshold]
    save_conceptcat_graph(resfile('concepts_99.svg'), labelcat_list)
    dump_json_file(resfile('report.json'), dict(
            header=dict(
                name='%s %s %s' % (MODEL, DATASET, seg),
                image='concepts_99.svg'),
            units=[
                dict(image='image/unit%d.jpg' % u,
                    unit=u, iou=iou, label=label, cat=labelcat[1])
                for u, (concept, label, labelcat, iou)
                in enumerate(unit_label_99)])
            )
    copy_static_file('report.html', resfile('+report.html'))
    resfile.done();

def make_upfn(dataset, model, layername):
    '''Creates an upsampling function.'''
    convs, data_shape = None, None
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

if __name__ == '__main__':
    main()

