# New-style dissection experiment code.
import torch, argparse
from netdissect import pbar, nethook, renormalize, pidfile
from netdissect import tally, imgviz, imgsave
import setting
torch.backends.cudnn.benchmark = True
import torchvision.models as models
import torchvision.transforms as transforms
from netdissect import parallelfolder
import torch.nn as nn
from torchvision import ops
from dissect_experiment import make_upfn, save_conceptcat_graph, dump_json_file, copy_static_file

# configs for our network dissection
IM_SIZE = 512
quantile=0.01
miniou = 0.04
thumbsize=500
seg = "netpqc"

def parseargs():
    parser = argparse.ArgumentParser(description="Run network dissection on MobileNetV2 variants. Run this script from the root of the repo.")
    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa('-m', '--model-weights-path', default=None, help="Path to finetuned weights. If no path provided, defaults to pre-trained weights.")
    aa('-d', '--dataset', default='dancer', help="Directory name for image dataset placed in \"experiments/dataset\"")
    aa('-l', '--layer', default="conv_4_2", help="Unflattened layer format is features.layer_num. Unflattened is operation_layer_index")
    aa('--unflat', default=False, action=argparse.BooleanOptionalAction, help="Load flat or unflattened MobileNetV2")
    args = parser.parse_args()
    return args 

def main():
    data_dir = "experiment/dataset/"

    args = parseargs()
    args.model = "MobileNetV2"

    # Get string for results directory name and load model
    if args.unflat:
        model_type = "unflat"
        model = load_unflattened_nst_model(args.model_weights_path)
    else:
        model_type = "flat"
        model = load_flattened_nst_model(args.model_weights_path)

    layername = str(args.layer)
    resdir = 'results/%s-%s-%s' % (model_type, args.dataset, layername)
    resfile = pidfile.exclusive_dirfn(resdir)

    # get layer of interest
    model.retain_layer(layername)
    # load dataset
    dataset = load_nst_data(data_dir + args.dataset) 

    # set up remaining functions/arguments for network dissection
    upfn = make_upfn(args, dataset, model, layername)
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
                name='%s %s %s' % (model_type, args.dataset, seg),
                image='concepts_99.svg'),
            units=[
                dict(image='image/unit%d.jpg' % u,
                    unit=u, iou=iou, label=label, cat=labelcat[1])
                for u, (concept, label, labelcat, iou)
                in enumerate(unit_label_99)])
            )
    copy_static_file('report.html', resfile('+report.html'))
    resfile.done();

def load_nst_data(path_to_data):
    """Load our datatset.

    Provides transform from our impressionist dataset, required for `ParallelImageFolders`.
    """
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

def load_unflattened_nst_model(path_to_model=None):
    """Load unflattened MobileNetV2 model, wrap with instrumented model so we can do network dissection.
    
    If no path is provided, pytorch's pretrained weights are used.
    """
    model = models.mobilenet_v2(pretrained=True)
    if path_to_model is not None:
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(torch.load(path_to_model))
    model = nethook.InstrumentedModel(model).cuda().eval()
    return model

def get_name_type(layer):
    """Helper function for flattening. Gets name of layer types in MobileNetV2."""
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

def load_flattened_nst_model(path_to_model=None):
    """Load flattened MobileNetV2 model, wrap with instrumented model so we can do network dissection.
    
    If no path is provided, pytorch's pretrained weights are used.
    """
    cnn = models.mobilenet_v2(pretrained=True)
    if path_to_model is not None:
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

if __name__ == '__main__':
    main()
