## fast_neural_style.lua

The script `fast_neural_style.lua` runs a trained model on new images. It has
the following flags:

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
  
  
## webcam_demo.lua

The script `webcam_demo.lua` runs models off the video stream
from a webcam. It has the following flags:

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


## train.lua

The script `train.lua` trains new feedforward style transfer models.
It has the following flags:

**Model options**:
- `-arch`: String specifying the architecture to use. Architectures are
  specified as comma-separated strings. The architecture used in the paper is
  `c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3`. All internal
  convolutional layers are followed by a ReLU and either batch normalization
  or instance normalization.
  - `cXsY-Z`: A convolutional layer with a kernel size of `X`, a stride of `Y`,
    and `Z` filters.
  - `dX`: A downsampling convolutional layer with `X` filters, 3x3 kernels,
    and stride 2.
  - `RX`: A residual block with two convolutional layers and `X`
    filters per layer.
  - `uX`: An upsampling convolutional layer with `X` filters, 3x3 kernels,
    and stride 1/2.
- `-use_instance_norm`: 1 to use instance normalization or 0 to use batch
  normalization. Default is 1.
- `-padding_type`: What type of padding to use for convolutions in residual
  blocks. The following choices are available:
  - `zero`: Normal zero padding everywhere.
  - `none`: No padding for convolutions in residual blocks.
  - `reflect`: Spatial reflection padding for all convolutions in residual blocks.
  - `replicate`: Spatial replication padding for all convolutions in residual
    blocks.
  - `reflect-start` (default): Spatial reflection padding at the beginning of
    the model and no padding for convolutions in residual blocks.
- `-tanh_constant`: There is a tanh nonlinearity after the final convolutional
  layer; this puts outputs in the range [-1, 1]. Outputs are then multiplied by
  the `-tanh_constant` so the outputs are in a more standard image range.
- `-preprocessing`: What type of preprocessing and deprocessing to use; either
  `vgg` or `resnet`. Default is `vgg`. If you want to use a ResNet as loss
  network you should set this to `resnet`.

**Loss options**:
- `-loss_network`: Path to a `.t7` file containing a pretrained CNN to be used
  as a loss network. The default is VGG-16, but the code should support many
  models such as VGG-19 and ResNets.
- `-content_layers`: Which layers of the loss network to use for the content
  reconstruction loss. This will usually be a comma-separated list of integers,
  but for complicated loss networks like ResNets it can be a list of
  of [layer strings](https://github.com/jcjohnson/neuralstyle2/blob/master/neuralstyle2/layer_utils.lua#L3).
- `-content_weights`: Weights to use for each content reconstruction loss.
  This can either be a single number, in which case the same weight is used for
  all content reconstruction terms, or it can be a comma-separated list of
  real values of the same length as `-content_layers`.
- `-style_image`: Path to the style image to use.
- `-style_image_size`: Before computing the style loss targets, the style image
  will be resized so its smaller side is this many pixels long. This can have a
  big effect on the types of features transferred from the style image.
- `-style_layers`: Which layers of the loss network to use for the style
  reconstruction loss. This is a comma-separated list of the same format as
  `-content_layers`.
- `-style_weights`: Weights to use for style reconstruction terms. Either a
  single number, in which case the same weight is used for all style
  reconstruction terms, or a comma-separated list of weights of the same length
  as `-style_layers`.
- `-style_target_type`: What type of style targets to use; either `gram` or
  `mean`. Default is `gram`, in which case style targets are Gram matrices as
  described by Gatys et al. If this is `mean` then the spatial average will be
  used as a style target instead of a Gram matrix.
- `-tv_strength`: Strength for total variation regularization on the output
  of the transformation network. Default is `1e-6`; higher values encourage
  the network to produce outputs that are spatially smooth.

**Training options**:
- `-h5_file`: HDF5 dataset created with `scripts/make_style_dataset.py`.
- `-num_iterations`: The number of gradient descent iterations to run.
- `-max_train`: The maximum number of training images to use; default is -1
  which uses the entire training set from the HDF5 dataset.
- `-batch_size`: The number of content images per minibatch. Default is 4.
- `-learning_rate`: Learning rate to use for Adam. Default is `1e-3`.
- `-lr_decay_every`, `-lr_decay_after`: Learning rate decay. After every
  `-lr_decay_every` iterations the learning rate is multiplied by
  `-lr_decay_factor`. Setting `-lr_decay_every` to -1 disables learning rate decay.
- `-weight_decay`: L2 regularization strength on the weights of the
  transformation network. Default is 0 (no L2 regularization).

**Checkpointing**:
- `-checkpoint_every`: Every `-checkpoint_every` iterations, check performance
  on the validation set and save both a `.t7` model checkpoint and a `.json`
  checkpoint with loss history.
- `-checkpoint_name`: Path where checkpoints are saved. Default is `checkpoint`,
  meaining that every `-checkpoint_every` iterations we will write files
  `checkpoint.t7` and `checkpoint.json`.

**Backend**:
- `-gpu`: Which GPU to use; default is 0. Set this to -1 to train in CPU mode.
- `-backend`: Which backend to use for GPU, either `cuda` or `opencl`.
- `-use_cudnn`: Whether to use cuDNN when using CUDA; 0 for no and 1 for yes.

## slow_neural_style.lua

The script `slow_neural_style.lua` uses the optimization-based style transfer method
similar to the original [neural-style](https://github.com/jcjohnson/neural-style).

It has the following flags:

**Basic Options**
- `-content_image`: Path to the content image to use.
- `-style_image`: Path to the style image to use.
- `-image_size`: Size of the generated image; its longest side is this many pixels long.

**Output Options**
- `-output_image`: Path where the output image will be written.
- `-print_every`: Losses will be printed after every `-print_every` iterations.
- `-save_every`: Images will be written every `-save_ever` iterations.

**Loss options**
All of these flags are the same as those in `train.lua`:
- `-loss_network`: Path to a `.t7` file containing a pretrained CNN to be used
  as a loss network. The default is VGG-16, but the code should support many
  models such as VGG-19 and ResNets.
- `-content_layers`: Which layers of the loss network to use for the content
  reconstruction loss. This will usually be a comma-separated list of integers,
  but for complicated loss networks like ResNets it can be a list of
  of [layer strings](https://github.com/jcjohnson/neuralstyle2/blob/master/neuralstyle2/layer_utils.lua#L3).
- `-content_weights`: Weights to use for each content reconstruction loss.
  This can either be a single number, in which case the same weight is used for
  all content reconstruction terms, or it can be a comma-separated list of
  real values of the same length as `-content_layers`.
- `-style_image_size`: Before computing the style loss targets, the style image
  will be resized so its smaller side is this many pixels long. This can have a
  big effect on the types of features transferred from the style image.
- `-style_layers`: Which layers of the loss network to use for the style
  reconstruction loss. This is a comma-separated list of the same format as
  `-content_layers`.
- `-style_weights`: Weights to use for style reconstruction terms. Either a
  single number, in which case the same weight is used for all style
  reconstruction terms, or a comma-separated list of weights of the same length
  as `-style_layers`.
- `-style_target_type`: What type of style targets to use; either `gram` or
  `mean`. Default is `gram`, in which case style targets are Gram matrices as
  described by Gatys et al. If this is `mean` then the spatial average will be
  used as a style target instead of a Gram matrix.
- `-tv_strength`: Strength for total variation regularization on the output
  of the transformation network. Default is `1e-6`; higher values encourage
  the network to produce outputs that are spatially smooth.
  
**Optimization Options**
- `-learning_rate`: Learning rate to use for optimization
- `-optimizer`: Either `lbfgs`, `adam`, or any other method from [torch/optim](https://github.com/torch/optim).
- `-num_iterations`: Number of iterations to run for

**Backend Options**:
- `-gpu`: Which GPU to use; default is 0. Set this to -1 to train in CPU mode.
- `-backend`: Which backend to use for GPU, either `cuda` or `opencl`.
- `-use_cudnn`: Whether to use cuDNN when using CUDA; 0 for no and 1 for yes.

**Other Options**
- `-preprocessing`: What type of preprocessing and deprocessing to use; either
  `vgg` or `resnet`. Default is `vgg`. If you want to use a ResNet as loss
  network you should set this to `resnet`.
