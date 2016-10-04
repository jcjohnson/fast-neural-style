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

The full set of options for this script are [described here](flags.md#trainlua).

