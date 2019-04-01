![Python](https://img.shields.io/badge/python-v3.6.3-blue.svg)
![Tensorflow-GPU](https://img.shields.io/badge/tensorflow-v1.7.0-blue.svg)
![Keras](https://img.shields.io/badge/keras-v2.1.5-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-v1.0.0-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-v9.1.0-green.svg)
![cudNN](https://img.shields.io/badge/cuDNN-v7.1.2-green.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

# Tensorflow Speech Recognition

## Intro
A complete walk-through on how to train deep learning models for Google Brain's 
Tensorflow Speech Recognition challenge on [Kaggle.](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge "Kaggle & Google Brain's Speech Recognition Challenge site")

Consists of Jupyter notebooks that can be sequentially run on the raw data 
provided by the creators of the challenge, as well as both **keras** and 
**tensorflow** scripts to train convolutional machine learning models on the 
preprocessed data.

Developed & tested on a Paperspace cloud instance with an NVIDIA Quadro P6000
(Pascal generation) graphics card, with an Ubuntu 16.04 (Xenial) OS.

This includes, in sequential order:

1) Splitting the raw data into balanced data sets ![Jupyter Notebooks Logo](https://i.ibb.co/KxLnVRY/jupyter-logo-small.png "Jupyter Notebook Logo")
2) Data investigation and visualization ![Jupyter Notebooks Logo](https://i.ibb.co/KxLnVRY/jupyter-logo-small.png "Jupyter Notebook Logo")
3) Preprocessing and data augmentation ![Jupyter Notebooks Logo](https://i.ibb.co/KxLnVRY/jupyter-logo-small.png "Jupyter Notebook Logo")
4) Model experiments on a small sample set ![Jupyter Notebooks Logo](https://i.ibb.co/KxLnVRY/jupyter-logo-small.png "Jupyter Notebook Logo")
5) Three ways to train the final models with Jupyter, Keras and Tensorflow ![Jupyter Notebooks Logo](https://i.ibb.co/KxLnVRY/jupyter-logo-small.png "Jupyter Notebook Logo") ![Keras Logo](https://i.ibb.co/NVBJ83Z/keras-logo-small.png "Keras Logo") ![Tensorflow Logo](https://i.ibb.co/fdksYmS/tensorflow-logo-small.png "Tensorflow Logo")

## Table of Contents
- [Intro](#intro)
- [Setup](#setup)
  - [Data Source](#data-soruce)
  - [Hardware](#hardware)
  - [Software](#software)
    - [Virtual Environment](#virtual-environment)
    - [Requirements](#requirements)
    - [CUDA and cudNN](#cuda-and-cudnn)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [License](#license)

## Setup
This section will guide you through setting up everything that you'll need to
follow along. 

This includes obtaining the raw data from the Kaggle [data source](#data-source), checking 
[hardware](#hardware) requirements and installing the proper [software](#software) 
libraries, in your preferred virtual environment.

### Data Source
In order to participate in Kaggle challenges you need to register an account 
on their website. Once you have a verified username and password, you can 
install their [kaggle cli](https://github.com/Kaggle/kaggle-api "Kaggle Cli Repository on Github")
tool and use the following command to download the original raw data:

    kaggle competitions download -c tensorflow-speech-recognition-challenge
    
Alternatively, if you're not interested in the preprocessing and just want to
jump straight to training models, you can download the processed bcolz 
files from [the internet archive](https://archive.org/details/voice_data "Internet Archive link with processed bcolz files").

### Hardware
I recommend using a GPU machine. On pure CPU the training code may 
take hours to finish. However, if you're only interested in the preprocessing 
part, a CPU should suffice.

Below are my **personal opinions** on viable options for individual ML 
enthusiasts, none of the below-mentioned providers are sponsoring me in any 
way. I've also provided links to useful articles about the different options
in the [References](#references) section (sources [1] to [7]).
 
There are good options available from cloud providers like Paperspace or Amazon
Web Services, which I have personal experience with, or you can look into 
Azure, Floydhub, Crestle, Hetzner and Google Cloud which have been 
recommended to me by other ML engineers. 

If you choose to go with a cloud-based solution, you may consider the 
dedicated Paperspace instances with these Pascal generation GPUs (costs from 
early 2019):

1. NVIDIA Quadro P6000 (24GB GPU memory) for $1.10/h (the version on which 
this repo was developed)
2. NVIDIA Quadro P5000 (16GB GPU memory) for $0.78/h
3. V100 (Volta architecture, 16GB GPU memory) for $2.30/h

The AWS recommended alternative would be the p3.xlarge instances with Volta 
generation GPUs, namely the Tesla V100. A more affordable alternative could 
be p2.xlarge instances with a Tesla K80, from an older generation.

If you choose to build your own box or have one available to you, I can 
personally recommend the GTX 1080 Ti for $700. 

To figure out what GPU is available on your machine, you can e.g. run:

    glxinfo | grep OpenGL
 
### Software

This was developed and tested on Python 3.6.3.

#### Virtual Environment

I recommend using the **venv** python module with pip for a virtual environment
or the package, dependency and environment management system such as **conda**. 

To install the venv library via pip:

    pip install virtualenv
    
To create your custom virtual environment:

    python3 -m venv /path/to/your/virtual/environment
    
You can then activate that virtual environment on Mac and Linux via:

    source <venv>/bin/activate
    
And on Windows via:

    <venv>\Scripts\activate
    
where <venv> is your custom virtual environment path.

#### Requirements

You can easily install all required python libraries by using the **req.txt** 
file, within your activated virtual environment:

    pip install -r req.txt
    
Or, if you're using conda:

    conda install --file req.txt 

#### CUDA and cudNN

You may also want to double-check that you've got the right versions of the 
NVIDIA software.

To check the version of CUDA you can use:

    nvcc --version
    
To check the version of cuDNN, you first need to find the `cudnn.h` file:

    whereis cudnn.h
    
And then to actually check the version, depending on the location use:

    ﻿cat /some/location/cudnn.h | grep CUDNN_MAJOR -A 2

If you don't have CUDA installed, you can download it from the NVIDIA 
[CUDA webpage](https://developer.nvidia.com/cuda-90-download-archive "Nvidia's CUDA download address")
and follow the installation instructions provided by them, per OS.

## Usage
WIP
### Data Processing
WIP
### Model Training
WIP
## Acknowledgements
## References
Original Kaggle challenge:
- [0] https://www.kaggle.com/c/tensorflow-speech-recognition-challenge

On GPU machines, both in the cloud and build-a-box:

- [1] http://forums.fast.ai/t/cost-effectiveness-of-the-different-cloud-server-renting-options/7300
- [2] https://medium.com/initialized-capital/benchmarking-tensorflow-performance-and-cost-across-different-gpu-options-69bd85fe5d58
- [3] https://www.paperspace.com/pricing
- [4] https://rare-technologies.com/machine-learning-benchmarks-hardware-providers-gpu-part-2/
- [5] https://towardsdatascience.com/how-the-f-does-nvidia-name-gpus-aef9c684362a
- [6] https://aws.amazon.com/ec2/instance-types/
- [7] https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415
- [8] https://www.ec2instances.info/?region=eu-west-1
- [9] https://aws.amazon.com/ec2/pricing/on-demand/
## License