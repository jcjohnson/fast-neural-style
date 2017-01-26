require 'torch'
require 'nn'
require 'image'
require 'riseml'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'


--[[
Use a trained feedforward model to stylize either a single image or an entire
directory of images.
--]]

local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', 'models/instance_norm/candy.t7')
cmd:option('-image_size', 320)
cmd:option('-median_filter', 0)
cmd:option('-timing', 0)

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)


local function main()
  local opt = cmd:parse(arg)

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local ok, checkpoint = pcall(function() return torch.load(opt.model) end)
  if not ok then
    print('ERROR: Could not load model from ' .. opt.model)
    print('You may need to download the pretrained models by running')
    print('bash models/download_style_transfer_models.sh')
    return
  end
  local model = checkpoint.model
  model:evaluate()
  model:type(dtype)
  if use_cudnn then
    cudnn.convert(model, cudnn)
    if opt.cudnn_benchmark == 0 then
      cudnn.benchmark = false
      cudnn.fastest = true
    end
  end

  local preprocess_method = checkpoint.opt.preprocessing or 'vgg'
  local preprocess = preprocess[preprocess_method]

  local function run_image(img_data)
    local byte_tensor = torch.ByteTensor(torch.ByteStorage():string(img_data))
    local img = image.decompressJPG(byte_tensor, 3)
    if opt.image_size > 0 then
      img = image.scale(img, opt.image_size)
    end
    local H, W = img:size(2), img:size(3)

    local img_pre = preprocess.preprocess(img:view(1, 3, H, W)):type(dtype)
    local timer = nil
    if opt.timing == 1 then
      -- Do an extra forward pass to warm up memory and cuDNN
      model:forward(img_pre)
      timer = torch.Timer()
      if cutorch then cutorch.synchronize() end
    end
    local img_out = model:forward(img_pre)
    if opt.timing == 1 then
      if cutorch then cutorch.synchronize() end
      local time = timer:time().real
      print(string.format('Image %s (%d x %d) took %f',
            in_path, H, W, time))
    end
    local img_out = preprocess.deprocess(img_out)[1]

    if opt.median_filter > 0 then
      img_out = utils.median_filter(img_out, opt.median_filter)
    end
    return image.compressJPG(img_out):storage():string()
  end

  riseml.serve(run_image)
end
main()
