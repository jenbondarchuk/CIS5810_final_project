# CIS5810 Final Project Group 16 - AutoArt

This is the network dissect code used to generate results on our variants of `MobileNetV2`.
This code is updated from [David Bau's Network Dissection Repo](https://github.com/davidbau/dissect).

The scripts in this repo expect a cuda-enabled GPU and was tested with CUDA 12.3 on Ubuntu 22.04. `Python 3.10` was used.

If these requirements are met, run `pip install -r requirements.txt` from this root of this repo to install all necessary pip packges.

To run network dissection for NST:
* Place an image dataset in `experiment/dataset/<your dataset dir>/<images>`.
* Download the weights for the fine-tuned model.
    * These are available in the project google drive, or in the project zip as `mobilenetv2_finetuned.pt`.
* From the root of this repo, run `python experiment/nst.py <path to model weights> <dataset directory name> <layer_name> <flat>`.
    * If you're not sure of the arguments to enter, run `python experiment/nst.py --help`.
* Results will be in the root of this repo.
    * See `results/<your results>/image/` for the images used in our presentation.