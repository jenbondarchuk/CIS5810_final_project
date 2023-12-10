# Notes from getting the 2020 repo to run
**These updates are already included in this project repo**
1. Ignore setup stuff, just install `pytorch torchvision torchaudio matplotlib statsmodels scipy` etc
1. Copy `netdissect/` into `experiments/`
1. Remove relative imports (i.e. `from . import settings` should be `import settings`)
1. Add ssl ignore block in `experiment/setting.py` because their ssl cert is expired
    ```python
    import ssl

      try:
          _create_unverified_https_context = ssl._create_unverified_context
      except AttributeError:
          # Legacy Python that doesn't verify HTTPS certificates by default
          pass
      else:
          # Handle target environment that doesn't support HTTPS verification
    ```
1. Run `pip install Ninja`
1. We need to update some stuff in `dissect/experiment/netdissect/upsegmodel/prroi_pool/src/prroi_pooling_gpu.c`:
    * Replace all `THCudaCheck` with `AT_CUDA_CHECK` (from [here](https://github.com/CoinCheung/pytorch-loss/pull/37))
    * Replace `#include <THC/THC.h>` with `#include <ATen/cuda/CUDAEvent.h>` (from [here](https://stackoverflow.com/a/72990619))
1. Run `python dissect_experiment.py`

Running notes:
* Takes a few minutes to download things initially, then it quickly gets to "Loading weights for net_decoder"
* VGG16 took 3 hours to run

My paths:  
dataset: "/home/jen/Documents/git/dissect/experiment/datasets/dancer"  
finetuned weights: "/home/jen/Downloads/mobilenetv2_finetuned.pt"