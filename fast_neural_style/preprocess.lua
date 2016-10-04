require 'torch'


local M = {}


local function check_input(img)
  assert(img:dim() == 4, 'img must be N x C x H x W')
  assert(img:size(2) == 3, 'img must have three channels') 
end


M.resnet = {}

local resnet_mean = {0.485, 0.456, 0.406}
local resnet_std = {0.229, 0.224, 0.225}


--[[
Preprocess an image before passing to a ResNet model. The preprocessing is easy:
we just need to subtract the mean and divide by the standard deviation. These
constants are taken from fb.resnet.torch:

https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

Input:
- img: Tensor of shape (N, C, H, W) giving a batch of images. Images are RGB
  in the range [0, 1].
]]
function M.resnet.preprocess(img)
  check_input(img)
  local mean = img.new(resnet_mean):view(1, 3, 1, 1):expandAs(img)
  local std = img.new(resnet_std):view(1, 3, 1, 1):expandAs(img)
  return (img - mean):cdiv(std)
end

-- Undo ResNet preprocessing.
function M.resnet.deprocess(img)
  check_input(img)
  local mean = img.new(resnet_mean):view(1, 3, 1, 1):expandAs(img)
  local std = img.new(resnet_std):view(1, 3, 1, 1):expandAs(img)
  return torch.cmul(img, std):add(mean)
end


M.vgg = {}

local vgg_mean = {103.939, 116.779, 123.68}

--[[
Preprocess an image before passing to a VGG model. We need to rescale from
[0, 1] to [0, 255], convert from RGB to BGR, and subtract the mean.

Input:
- img: Tensor of shape (N, C, H, W) giving a batch of images. Images 
]]
function M.vgg.preprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return img:index(2, perm):mul(255):add(-1, mean)
end


-- Undo VGG preprocessing
function M.vgg.deprocess(img)
  check_input(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(2, perm)
end


return M
