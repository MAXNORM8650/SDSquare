# AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models using Simplex Noise

This is the github repository for an anomaly detection approach utilising DDPMs with simplex noise implemented in
pytorch.

The code was written by [Komal Kumar](https://github.com/MAXNORM8650) and is based on
the [Guided Diffusion Repo](https://github.com/Julian-Wyatt/AnoDDPM), [Predictive Convolutional Attentive block repo](https://github.com/ristea/sspcab), and [guided-diffusion repo](https://github.com/openai/guided-diffusion).

![model pdf](https://user-images.githubusercontent.com/97806194/225223535-0ae2064d-1a00-4649-83af-353edc7d58b5.png)

![SD drawio](https://user-images.githubusercontent.com/97806194/225223942-f6800cad-63c4-42d6-a40d-cf55e6a450e2.png)


## File structure:

- dataset.py - custom dataset loader
- detection.py - code for generating measures & initial testing and experimentation.
- diffusion_training.py - training procedure
- evaluation.py - functions for measures and metrics
- GaussianDiffusion.py - Gaussian architecture with custom detection, forked from https://github.
  com/openai/guided-diffusion
- generate_images.py - generates images for Figs in paper
- helpers.py - helper functions for use in several places ie checkpoint loading
- simplex.py - Simplex class - forked from https://github.com/lmas/opensimplex with added multi-scale code.
- UNet.py - UNet architecture, forked from https://github.com/openai/guided-diffusion
- test_args/args{i}.json - primary example seen below
- model/diff-params-ARGS={i}/params-final.pt - checkpoint for i'th arg
- Examples/ - demonstration of early testing
- diffusion-videos/ARGS={i}/ - video outputs of varying args across training, testing and detection
- diffusion-training-images/ARGS={i}/ - detection images
- metrics/ - storage of varying metrics
- final-outputs/ - outputs from generate_images.py

### Train

To train a model, run `python3 diffusion_training.py ARG_NUM` where `ARG_NUM` is the number relating to the json arg
file. These arguments are stored in ./test_args/ and are called args38.json for example.

### Evaluate

To evaluate a model, run `python detection.py ARG_NUM`, and ensure the script runs the correct sub function.

### Datasets
BRATS2021 and Pneumonia X-Ray both publicly available datasets. Please refer to the paper for more details.
## Example args:

{
  "img_size": [
    256,
    256
  ],
  "Batch_Size": 1,
  "EPOCHS": 4000,
  "T": 1000,
  "base_channels": 128,
  "channels": 3,
  "beta_schedule": "cosine",
  "channel_mults": "",
  "loss-type": "l2",
  "loss_weight": "none",
  "train_start": true,
  "lr": 1e-4,
  "random_slice": false,
  "sample_distance": 800,
  "weight_decay": 0.0,
  "save_imgs": true,
  "save_vids": false,
  "dropout": 0,
  "attention_resolutions": "32,16,8",
  "num_heads": 2,
  "num_head_channels": -1,
  "noise_fn": "simplex",
  "dataset": "pneumonia"
}
