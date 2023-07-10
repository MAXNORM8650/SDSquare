# SDSquare: Simplex Diffusion and Selective Denoising Self-Supervised Model for Anomaly Detection in Medical Images

This is the github repository for an anomaly detection approach utilising DDPMs with simplex noise implemented in
pytorch.

The code was written by [Komal Kumar](https://github.com/MAXNORM8650) and is based on
the [Denoising diffusion model for Anomaly detection repo](https://github.com/Julian-Wyatt/AnoDDPM), [Predictive Convolutional Attentive block repo](https://github.com/ristea/sspcab), and [guided-diffusion repo](https://github.com/openai/guided-diffusion).
#	Simplex Diffusion and Selective Denoising (SD)Square
![model pdf](https://user-images.githubusercontent.com/97806194/225223535-0ae2064d-1a00-4649-83af-353edc7d58b5.png)
# Selective denoising backward process
![SD drawio](https://user-images.githubusercontent.com/97806194/225223942-f6800cad-63c4-42d6-a40d-cf55e6a450e2.png)


## Main file structure:

This repository contains code for a project focused on diffusion models and detection. The project includes various components and functionalities for training, evaluation, and experimentation with these models. Below is a brief overview of the main files and directories in this repository:

--dataset.py: This file provides a custom dataset loader, allowing you to load and preprocess your own dataset for training and testing.

detection.py: It contains code for generating measures, performing initial testing, and conducting experiments related to detection.

diffusion_training.py: This file implements the training procedure for your diffusion model, allowing you to train the model using your dataset.

evaluation.py: It includes functions for measuring and evaluating the performance of your diffusion model on different metrics and evaluation criteria.

GaussianDiffusion.py: This file represents a Gaussian architecture with custom detection capabilities. It is based on a forked version of the original code from the OpenAI repository.

helpers.py: It provides various helper functions that can be utilized in different parts of the project, such as loading checkpoints and handling common operations.

simplex.py: This file contains a class named Simplex that offers noise generation functionality. It is a forked version of the opensimplex library, with additional support for 3D and 4D noise generation for colored applications.

UNet.py: This file implements the UNet architecture, which is forked from the OpenAI repository. It includes an added selective-denoising block, enhancing the capabilities of the model.

In addition, the repository includes the following directories:

test_args/args{i}.json: These JSON files provide example arguments for testing and configuring different aspects of the project.

model/diff-params-ARGS={i}/params-final.pt: These files represent checkpoints saved during the training process. They can be used to restore and utilize trained models for further experimentation or deployment.

diffusion-videos/ARGS={i}/: This directory contains video outputs generated during various stages of training, testing, and detection. These videos can be helpful for visualizing the performance and progress of the models.

diffusion-training-images/ARGS={i}/: This directory contains detection images related to the project. These images can be used to showcase the results and illustrate the effectiveness of the detection algorithms implemented in the project.

Feel free to modify and customize this description based on your project's specific details and goals.
### Train

To train a model, run `python .\diffusion_training.py argsN` where `argsN` is the number relating to the json arg
file. These arguments are stored in ./test_args/ and are called args38.json for example.

### Evaluate

To evaluate a model, run `python .\detection.py argsN`, and ensure the script runs the correct sub function.

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
