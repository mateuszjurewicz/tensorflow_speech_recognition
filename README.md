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
  - [Data Source](#data-source)
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
- [Contribute](#contribute)
- [License](#license)

## Setup
This section will guide you through setting up everything that you'll need to
follow along. 

This includes obtaining the raw data from the Kaggle [data source](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data "Link to Kaggle's data description for this challenge"), checking 
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

The [tensorflow GPU support](https://www.tensorflow.org/install/gpu "Tensorflow CUDA installation instructions page")
page can also be helpful in setting this up.

## Usage

This section will give you a step-by-step understanding of what's achieved by
each of the consecutive jupyter notebooks for the data preprocessing part of 
this challenge, followed by an explanation of the various machine learning 
model training scripts in jupyter + keras, pure keras and pure tensorflow. 

### Data Processing

What we get from Kaggle in terms of training data is a set of .wav files, 
split into folders named after one of 30 categories. Some of these are 
mislabelled. Their name also contains and identifier of the person who said 
them, which will be important when balancing our dataset.

We are only meant to predict 12 categories, with the remaining 
ones becoming classified as _unknown_. There's also a _background_noise_ 
folder containing longer samples, which we can use to obtain examples 
belonging to the _silence_ category.

##### 0. Separation for training, validation and sample

In the [zeroth notebook](https://github.com/mateuszjurewicz/tensorflow_speech_recognition/blob/master/0.%20Separation%20for%20training%2C%20validation%20and%20sample.ipynb "Link to the zeroth jupyter notebook")
we use the provided testing_list.txt and validation_list.txt lists to split the 
original train set into main/test, main/cv and main/train (by moving files). 

My hope is that this notebook will make you more comfortable with using the 
**glob** library in order to manipulate the file structure of your data.

This has the  benefit of putting files recorded by the same person in only 
one subset, so the model can't latch onto a person's voice characteristics. 
This ensures that there's **no data leakage**. Otherwise the model could 
learn that a specific person's voice characteristics are always tied to one 
category, leading to overfitting.

We're also preparing a smaller **sample set**, further separated into 
the expected train, test & validation subsets. The point of this is to be 
able to test your code on a smaller slice of data and see if it runs to 
completion, without expecting the models to achieve extremely high precision 
or recall. Depending on its size, it can also allow us to iterate on 
different model architectures quicker. In my professional experience, human time
is often the scarcest resource, so using sample sets is always recommended.

Finally, we also use the longer background noise files by cutting them down 
into 1 second long .wav files, similar in length to the examples from other 
categories.

The end result is a file structure of the main and sample data, further 
divided into train, test and cv subsets, in a way that should prevent data 
leakage. 

It is important to notice that in main/test we now have 250 examples per 
category, but in the case of _unknown_ we have over 4000 examples. In main/train
we have 1850 examples per category and 32K in unknown. This makes our 
main data set very unbalanced (no such problems for the sample set). We will 
handle this in the main model training notebook (the fourth one).

##### 1. Visualization and data investigation

In the [first notebook](https://github.com/mateuszjurewicz/tensorflow_speech_recognition/blob/master/1.%20Visualization%20and%20data%20investigation.ipynb "Link to the first notebook")
we'll be visualising our data as waveforms, spectrograms and chromagrams.

It's often useful to get a set of eyes on the data you're working with. Doing
this early in the process has many times allowed me to catch something that 
would be important later on. In this case, we'll also want to listen to our 
examples.

Looking at the images obtained by applying these kinds of transformations to 
our sound samples can give us an idea of the features e.g. convolutional 
layers of our models might be able to latch onto. 


##### 2. Preprocessing and data augmentation

In the [second notebook](https://github.com/mateuszjurewicz/tensorflow_speech_recognition/blob/master/2.%20Preprocessing%20and%20data%20augmentation.ipynb "Link to the second notebook")

This notebook's goal is to use various preprocessing / data augmentation 
techniques that are known to work well on audio data, in order to obtain and 
eventually persist these preprocessed subsets. 

This way we can have a lot of different possible inputs when we test 
various model architectures later on, without the need to do this 
preprocessing "live", every time we load a new batch of data. The alternative
is to embed the preprocessing into our data _generators_, in python terms.

In terms of preprocessing we are turning our audio files into:
1. MFCCs (MEL Frequency Cepstrum Coefficients)
2. Mel spectrograms
3. FFT (apply the Fast Fourier Transform)
4. Tempograms
    
We are also augmenting the data through adding varying levels of background 
noise to them, shifting the actual word utterance's position to later or 
earlier within the recording and finally stretching (increasing the wavelength),
to mimic differences in the timbre of people's voices.

Some of these methods can be stacked together, leading to a lot of possible 
combinations. Ultimately the best preprocessing method will be the one that 
leads to the best models.

##### 3. Model experiments - sample set

In the [third notebook](https://github.com/mateuszjurewicz/tensorflow_speech_recognition/blob/master/3.%20Model%20experiments%20-%20sample%20set.ipynb "Link to the third notebook")
we experiment with simple models to assess the effectiveness of our 
preprocessing & data augmentation techniques by training on the small sample 
set.

The assumption is that techniques that worked well on a small section of data
will generalize well to the entire set. A valid criticism here is that we are
trying to estimate two things at the same time, one of which has an influence
on the other. Namely we are trying to choose a preferred preprocessing 
pipeline, whilst assessing model architectures by the models' performance.

We could exhaustively test every possible combination of the preprocessing 
methods on every model architecture, but this would be expensive in terms of 
human time. Nevertheless any possible bias that we are aware of should be 
clearly stated.

One of the first things we do is **one-hot encoding our y**, our target 
predictions. This simply means turning the category name into a vector of all
zeroes and a single one, placed at the index corresponding to the class.

We then proceed to preprocess and persist our data. After that we begin 
experimenting with simple models, bearing in mind the random guessing 
accuracy of 0.833% as the first threshold to beat.

As our metrics we use recall, precision and F1 score, as this is a multiclass
classification problem. The final Kaggle evaluation metric is multiclass 
accuracy.

The model architectures that are included are:

1. Linear Models
2. Random Forest Classifiers
3. Multilayer Perceptron
4. Simple, Fully-connected Neural Networks
5. Deep Convolutional Neural Networks
6. Gated Recurrent Unit Networks (GRU)
7. Long Short Term Memory Networks (LSTM)

The models are trained with different learning rates, solvers, regularization 
and batch normalization. Hyperparameters have been tuned to maximize 
performance. 

The best performance has been harnessed from deep convolutional architectures
on variously preprocessed data and thus we focused on them in the next notebook.

Certain more expensive architectures were also tested without a noticeable 
performance improvement (e.g. transfer learning through unfreezing varying 
number of top layers of e.g. ResNet50 on 2D transformed data), hence they 
were excluded.

##### 4. Model training - full set

In the [fourth notebook](https://github.com/mateuszjurewicz/tensorflow_speech_recognition/blob/master/4.%20Model%20training%20-%20full%20set.ipynb "Link to the fourth notebook")

### Model Training
WIP

## Acknowledgements

I would like to give a huge amount of thanks and recognition to the entire 
Kaggle community, which is always great at exchanging ideas, things that 
worked and, equally importantly, *the things that did not*, once the challenges 
are done. I believe that *that* is a crucially valuable stage of the process.

In particular, the relatively simple deep convolutional model that I settled 
on after trying more complex ensemble methods and e.g. a partially 
retrained version of Keras' ResNet50 on variously preprocessed 2D data, can 
be read about in this [kaggle discussion](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47618 "Kaggle discussion on 1D convolutional models")
thread.

I would therefore like to highlight [Sukjae Cho](https://www.kaggle.com/jandjenter),
[은주니(ttagu99)](https://www.kaggle.com/ttagu99) and [vinvinvin](https://www.kaggle.com/vinvinvin).

Additionally, I would like to give special thanks to the entire FastAi 
community, particularly the 2 people who originally decided to make neural 
networks uncool again - [Jeremy Howard](https://www.fast.ai/about/#jeremy) and [Rachel Thomas](https://www.fast.ai/about/#rachel).

Since this is meant as a helpful guide for people with various levels of 
experience in the field, I would also like to point the beginners to a great 
introductory [Stanford ML course](https://www.coursera.org/learn/machine-learning "Coursera ML 101 course") by Andrew Ng, 
available on Coursera, which helped me tremendously back when I was beginning
my adventures with machine learning and data science. 

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

Papers with code, always a great thing and this needs to become a standard:

- [10] https://paperswithcode.com/sota

## Contribute

Contributions are always welcome! 

If you find an error, a false assumption in the rationale behind the data 
processing, a good way to improve the overall performance of the code or the
models - please reach out via email at the address given in my github bio or
propose a fix by opening a PR, following the [contribution guidelines](contribute.md).
 
## License
![GNU 3](https://i.ibb.co/3RcsTNw/gnu-license-small.png "GNU Copyleft logo")

To the extent possible under law, the author has waived all copyright and 
related or neighboring rights to this work.