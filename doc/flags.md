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
