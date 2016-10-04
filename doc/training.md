## Training new models

To train new style transfer models, first use the script
`scripts/make_style_dataset.py` to create an HDF5 file from folders of images.
You will then use the script `train.lua` to actually train models.

### Step 1: Prepare a dataset

You first need to install the header files for Python 2.7 and HDF5. On Ubuntu
you should be able to do the following:

```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
```

You can then install Python dependencies into a virtual environment:

```bash
virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment
```

With the virtual environment activated, you can use the script
`scripts/make_style_dataset.py` to create an HDF5 file from a directory of
training images and a directory of validation images:

```bash
python scripts/make_style_dataset.py \
  --train_dir path/to/training/images \
  --val_dir path/to/validation/images \
  --output_file path/to/output/file.h5
```

All models in this
repository were trained using the images from the
[COCO dataset](http://mscoco.org/).

The preprocessing script has the following flags:
- `--train_dir`: Path to a directory of training images.
- `--val_dir`: Path to a directory of validation images.
- `--output_file`: HDF5 file where output will be written.
- `--height`, `--width`: All images will be resized to this size.
- `--max_images`: The maximum number of images to use for training
  and validation; -1 means use all images in the directories.
- `--num_workers`: The number of threads to use.

### Step 2: Train a model

After creating an HDF5 dataset file, you can use the script `train.lua` to
train feedforward style transfer models. First you need to download a
Torch version of the
[VGG-16 model](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)
by running the script

```bash
bash models/download_vgg16.sh
```

This will download the file `vgg16.t7` (528 MB) to the `models` directory.

You will also need to install
[deepmind/torch-hdf5](https://github.com/deepmind/torch-hdf5)
which gives HDF5 bindings for Torch:

```
luarocks install https://raw.githubusercontent.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec
```

You can then train a model with the script `train.lua`. For basic usage the
command will look something like this:

```bash
th train.lua \
  -h5_file path/to/dataset.h5 \
  -style_image path/to/style/image.jpg \
  -style_image_size 384 \
  -content_weights 1.0 \
  -style_weights 5.0 \
  -checkpoint_name checkpoint \
  -gpu 0
```

Here are the full list of flags for `train.lua`:

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
  models such as VGG-16 and ResNets.
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

