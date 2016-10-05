require 'torch'
require 'nn'
require 'optim'
require 'image'

require 'fast_neural_style.PerceptualCriterion'
require 'fast_neural_style.TotalVariation'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'

--[[
Perform optimization-based style transfer as described in
--]]

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-content_image', 'images/content/chicago.jpg')
cmd:option('-style_image', 'images/styles/starry_night.jpg')
cmd:option('-image_size', 512)

-- Loss options
cmd:option('-loss_network', 'models/vgg16.t7')
cmd:option('-tv_strength', 1e-6)
cmd:option('-loss_type', 'L2', 'L2|SmoothL1')
cmd:option('-style_target_type', 'gram', 'gram|mean')

-- Options for content reconstruction
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')

-- Options for style reconstruction
cmd:option('-style_weights', '5.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_image_size', 512)

-- Options for DeepDream
cmd:option('-deepdream_layers', '')
cmd:option('-deepdream_weights', '')

-- Optimization
cmd:option('-learning_rate', 1.0)
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam')
cmd:option('-num_iterations', 500)

-- Output options
cmd:option('-output_image', 'out.png')
cmd:option('-print_every', 1)
cmd:option('-save_every', 50)

-- Other options
cmd:option('-preprocessing', 'vgg')

-- Backend options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda', 'cuda|opencl')
cmd:option('-use_cudnn', 1)


local opt = cmd:parse(arg)


local function main()
  if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
  end
  preprocess = preprocess[opt.preprocessing]
  
  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn)
  
  -- Set up the criterion
  local ok, loss_net = pcall(function() return torch.load(opt.loss_network) end)
  if not ok then
    print('ERROR: Could not load loss network from ' .. opt.loss_network)
    print('You may need to download the VGG-16 model by running:')
    print('bash models/download_vgg16.sh')
    return
  end
  print(loss_net)
  local style_layers, style_weights =
    utils.parse_layers(opt.style_layers, opt.style_weights)
  local content_layers, content_weights =
    utils.parse_layers(opt.content_layers, opt.content_weights)
  local deepdream_layers, deepdream_weights =
    utils.parse_layers(opt.deepdream_layers, opt.deepdream_weights)
  local crit_args = {
    cnn = loss_net,
    style_layers = style_layers,
    style_weights = style_weights,
    content_layers = content_layers,
    content_weights = content_weights,
    deepdream_layers = deepdream_layers,
    deepdream_weights = deepdream_weights,
    loss_type = opt.loss_type,
    agg_type = opt.style_target_type,
  }
  local crit = nn.PerceptualCriterion(crit_args):type(dtype)
  
  -- Set the content image
  local content_image = image.load(opt.content_image, 3)
  content_image = image.scale(content_image, opt.image_size)
  local H, W = content_image:size(2), content_image:size(3)
  content_image = preprocess.preprocess(content_image:view(1, 3, H, W))
  crit:setContentTarget(content_image:type(dtype))
  
  -- Set the style image
  local style_image = image.load(opt.style_image, 3)
  style_image = image.scale(style_image, opt.style_image_size)
  local H, W = style_image:size(2), style_image:size(3)
  style_image = preprocess.preprocess(style_image:view(1, 3, H, W))
  crit:setStyleTarget(style_image:type(dtype))

  -- Set up total variation regularization
  local tv = nn.Identity()
  if opt.tv_strength > 0 then
    tv = nn.TotalVariation(opt.tv_strength)
  end
  tv:type(dtype)

  local img = torch.randn(#content_image):type(dtype)
  
  -- Callback function for optim methods
  local f_calls = 0
  local function f(x)
    f_calls = f_calls + 1
    local tv_out = tv:forward(x)
    local loss = crit:forward(tv_out, {})
    local grad_tv_out = crit:updateGradInput(tv_out, {})
    local grad_x = tv:backward(x, grad_tv_out)
 
    if opt.print_every > 0 and f_calls % opt.print_every == 0 then
      print(string.format('Iteration %d, loss = %f', f_calls, loss))
    end
 
    if opt.save_every > 0 and f_calls % opt.save_every == 0 then
      local img_out = preprocess.deprocess(img:float())[1]
      local ext = paths.extname(opt.output_image)
      local basename = paths.basename(opt.output_image):split('%.')[1]
      local directory = paths.dirname(opt.output_image)
      local filename = string.format('%s/%s_%d.%s',
                          directory, basename, f_calls, ext)
      image.save(filename, img_out)
    end
    
    return loss, grad_x:view(-1)
  end

  if opt.optimizer == 'lbfgs' then
    local config = {
      maxIter=opt.num_iterations,
      learningRate=opt.learning_rate,
      verbose=true,
      tolX = 0,
    }
    optim.lbfgs(f, img, config)
  else
    local config = {
      learningRate = opt.learning_rate
    }
    local optim_fn = optim[opts.optimizer]
    if optim_fn == nil then
      error(string.format('Invalid optimizer "%s"', opts.optimizer))
    end
    for t = 1, opt.num_iterations do
      optim_fn(f, img, config)
    end
  end
end


main()
