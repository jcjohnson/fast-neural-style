require 'torch'
require 'hdf5'

local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  assert(opt.batch_size, 'Must provide batch size')
  self.preprocess_fn = preprocess[opt.preprocessing].preprocess

  self.task = opt.task

  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.batch_size = opt.batch_size
  
  self.split_idxs = {
    train = 1,
    val = 1,
  }
  
  self.image_paths = {
    train = '/train2014/images',
    val = '/val2014/images',
  }

  if opt.task == 'upsample' then
    local factor = opt.upsample_factor
    self.image_paths = {
      train = string.format('/train/x_%d', factor),
      val = string.format('/val/x_%d', factor),
    }
  end
  
  local train_size = self.h5_file:read(self.image_paths.train):dataspaceSize()
  self.split_sizes = {
    train = train_size[1],
    val = self.h5_file:read(self.image_paths.val):dataspaceSize()[1],
  }
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]

  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end

  if opt.task == 'upsample' then
    local size = self.h5_file:read('/train/y'):dataspaceSize()
    self.y_height = size[3]
    self.y_width = size[4]
  end
  
  if opt.max_train and opt.max_train > 0 then
    self.split_sizes.train = opt.max_train
  end

  self.rgb_to_gray = torch.FloatTensor{0.2989, 0.5870, 0.1140}
end


function DataLoader:reset(split)
  self.split_idxs[split] = 1
end


function DataLoader:getBatch(split)
  local path = self.image_paths[split]

  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
  
  -- Load images out of the HDF5 file
  local images = self.h5_file:read(path):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)

  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end

  -- Preprocess images
  images_pre = self.preprocess_fn(images)

  if self.task == 'upsample' then
    -- Also read high-res images out of the HDF5 file
    local y_path = string.format('/%s/y', split)
    local y_images = self.h5_file:read(y_path):partial(
                        {start_idx, end_idx},
                        {1, self.num_channels},
                        {1, self.y_height},
                        {1, self.y_width})

    local y_images_pre = self.preprocess_fn(y_images:float())
    
    return images_pre, y_images_pre
  elseif self.task == 'style' then
    -- For style transfer just return the images twice
    return images_pre, images_pre
  end
end

