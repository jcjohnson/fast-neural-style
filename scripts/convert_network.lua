require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'

local utils = require 'fast_neural_style.utils'

--[[
Convert model checkpoints to CPU-only .t7 checkpoints.

Input can be a caffe model (prototxt and caffemodel) in which case it will be
loaded using loadcaffe; input can also be a .t7 file with cudnn or CUDA layers.
]]

local cmd = torch.CmdLine()
cmd:option('-input_t7', '')
cmd:option('-input_prototxt', '')
cmd:option('-input_caffemodel', '')
cmd:option('-output_t7', '')
cmd:option('-clear_gradients', 1)

local function main()
  local opt = cmd:parse(arg)
  if (opt.input_t7 == '') == (opt.input_prototxt == '') then
    error('Must pass exactly one of -input_t7 or -input_prototxt')
  end
  if opt.output_t7 == '' then
    error('Must pass -output_t7')
  end
  local net = nil
  if opt.input_t7 ~= '' then
    print('Reading network from ' .. opt.input_t7)
    net = torch.load(opt.input_t7)
    if net.model then
      net = net.model
    end
  elseif opt.input_prototxt ~= '' then
    if opt.input_caffemodel == '' then
      error('Must pass -input_caffemodel with -input_prototxt')
    end
    print('Reading network from ' .. opt.input_caffemodel)
    net = loadcaffe.load(opt.input_prototxt, opt.input_caffemodel, 'nn')
  end
  cudnn.convert(net, nn)
  net:float()
  net:clearState()
  if opt.clear_gradients == 1 then
    utils.clear_gradients(net)
  end
  print('Saving network to ' .. opt.output_t7)
  torch.save(opt.output_t7, net)
end


main()
