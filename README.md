# fast-neural-style

This is the code for the paper

**Perceptual Losses for Real-Time Style Transfer and Super-Resolution**
<br>
[Justin Johnson](http://cs.stanford.edu/people/jcjohns/),
[Alexandre Alahi](http://web.stanford.edu/~alahi/),
[Li Fei-Fei](http://vision.stanford.edu/feifeili/)
<br>
To appear at [ECCV 2016](http://www.eccv2016.org/)

The paper builds on
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge by training
feedforward neural networks that apply artistic styles to images.
After training, our feedforward networks can stylize images up to **three orders
of magnitude faster** than the optimization-based method presented by Gatys et al.

This repository also includes an implementation of instance normalization as
described in the paper [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
by Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. This simple trick
significantly improves the quality of feedforward style transfer models.

Stylizing this image of the Stanford campus at a resolution of 1200x630
takes **50 milliseconds** on a Pascal Titan X:

<div align='center'>
  <img src='images/styles/candy.jpg' height="225px">
  <img src='images/content/hoovertowernight.jpg' height="225px">
  <img src='images/outputs/hoovertowernight_candy.jpg' height="346px">
</div>

In this repository we provide:
- The trained style transfer models used in the paper
- Additional models using instance normalization
- Code for running models on new images
- A demo that runs models in real-time off a webcam
- Code for training new feedforward style transfer models
- An implementation of optimization-based style transfer method described
  by Gatys et al.

If you find this code useful for your research, please cite

```
@inproceedings{Johnson2016Perceptual,
  title={Perceptual losses for real-time style transfer and super-resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  booktitle={European Conference on Computer Vision},
  year={2016}
}
```

## Pretrained Models
Download all pretrained style transfer models by running the script

```bash
bash models/download_style_transfer_models.sh
```

This will download ten model files (~200MB) to the folder `models/`.

## Models from the paper
The style transfer models we used in the paper will be located in the folder `models/eccv16`.
Here are some example results where we use these models to stylize this
image of the Chicago skyline with at an image size of 512:

<div align='center'>
  <img src='images/content/chicago.jpg' height="185px">
</div>
<img src='images/styles/starry_night_crop.jpg' height="155px">
<img src='images/styles/la_muse.jpg' height="155px">
<img src='images/styles/composition_vii.jpg' height='155px'>
<img src='images/styles/wave_crop.jpg' height='155px'>
<br>
<img src='images/outputs/eccv16/chicago_starry_night.jpg' height="142px">
<img src='images/outputs/eccv16/chicago_la_muse.jpg' height="142px">
<img src='images/outputs/eccv16/chicago_composition_vii.jpg' height="142px">
<img src='images/outputs/eccv16/chicago_wave.jpg' height="142px">

## Models with instance normalization
As discussed in the paper
[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
by Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky, replacing batch
normalization with instance normalization significantly improves the quality
of feedforward style transfer models.

We have trained several models with instance normalization; after downloading
pretrained models they will be in the folder `models/instance_norm`.

These models use the same architecture as those used in our paper, except with
half the number of filters per layer and with instance normalization instead of
batch normalization. Using narrower layers makes the models smaller and faster
without sacrificing model quality.

Here are some example outputs from these models, with an image size of 1024:

<div align='center'>
  <img src='images/styles/candy.jpg' height='174px'>
  <img src='images/outputs/chicago_candy.jpg' height="174px">
  <img src='images/outputs/chicago_udnie.jpg' height="174px">
  <img src='images/styles/udnie.jpg' height='174px'>
  <br>
  <img src='images/styles/the_scream.jpg' height='174px'>
  <img src='images/outputs/chicago_scream.jpg' height="174px">
  <img src='images/outputs/chicago_mosaic.jpg' height="174px">
  <img src='images/styles/mosaic.jpg' height='174px'>
  <br>
  <img src='images/styles/feathers.jpg' height='173px'>
  <img src='images/outputs/chicago_feathers.jpg' height="173px">
  <img src='images/outputs/chicago_muse.jpg' height="173px">
  <img src='images/styles/la_muse.jpg' height='173px'>
</div>

## Running on new images
Use the script `fast_neural_style.lua` to run a trained model on new images.
It has the following flags:

**Model options**:
- `-model`: Path to a `.t7` model file from `train.lua`.
- `-image_size`: Before being input to the network, images are resized so
  their longer side is this many pixels long. If 0 then use the original size
  of the image.
- `-median_filter`: If nonzero, use a
  [median filter](https://en.wikipedia.org/wiki/Median_filter) of this kernel
  size as a post-processing step. Default is 3.

**Input / Output**:
- `-input_image`: Path to a single image on which to run the model.
- `-input_dir`: Path to a directory of image files; the model will be run
  on all images in the directory.
- `-output_image`: When using `-input_image` to specify input, the
  path to which the stylized image will be written.
- `-output_dir`: When using `-input_dir` to specify input, this gives a path
  to a directory where stylized images will be written. Each output image
  will have the same filename as its corresponding input image.

**Backend options**:
- `-gpu`: Which GPU to use for processing (zero-indexed);
  use -1 to process on CPU.
- `-backend`: Which GPU backend to use; either `cuda` or `opencl`. Default is
  `cuda`; this is ignored in CPU mode.
- `-use_cudnn`: Whether to use cuDNN with the CUDA backend; 1 for yes and 0 for no.
  Ignored in CPU mode or when using the OpenCL backend. Default is 1.
- `-cudnn_benchmark`: Whether to use the cuDNN autotuner when running with cuDNN;
  1 for yes or 0 for no. Default is 0. If you want to run the model on many images
  of the same size, then setting this to 1 may give a speed boost.


## Webcam demo
You can use the script `webcam_demo.lua` to run one or more models in real-time
off a webcam stream. To run this demo you need to use `qlua` instead of `th`:

```bash
qlua webcam_demo.lua -models models/instance_norm/candy.t7 -gpu 0
```

With a Pascal Titan X you can easily run four models in realtime at 640x480:

<div align='center'>
  <img src='images/webcam.gif' width='700px'>
</div>

The webcam demo depends on a few extra Lua packages:
- [clementfarabet/lua---camera](https://github.com/clementfarabet/lua---camera)
- [torch/qtlua](https://github.com/torch/qtlua)

You can install / update these packages by running:

```bash
luarocks install camera
luarocks install qtlua
```

The webcam demo has the following flags:

Model options:
- `-models`: A comma-separated list of models to use.

Webcam options:
- `-webcam_idx`: Which webcam to use; default is 0.
- `-webcam_fps`: Frames per second to request from the webcam; default is 60.
- `-height`, `-width`: Image resolution to request from the webcam.

Backend options:
- `-gpu`: Which GPU to use (zero-indexed); use -1 for CPU. You will likely need
  a GPU to get good results.
- `-backend`: GPU backend to use, either `cuda` or `opencl`.
- `-use_cudnn`: Whether to use cuDNN when using CUDA; 1 for yes, 0 for no.


## Training new models

You can [find instructions for training new models here](doc/training.md).

