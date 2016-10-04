require 'torch'
require 'nn'
local utils = require 'utils'

local cmd = torch.CmdLine()
cmd:option('-net1', 'models/vgg16/vgg16.t7')
cmd:option('-net2', 'models/vgg16/vgg16_nograd.t7')
cmd:option('-gpu', 0)
local opt = cmd:parse(arg)

local dtype = utils.setup_gpu(opt.gpu)
local net1 = torch.load(opt.net1)
local net2 = torch.load(opt.net2)
net1:type(dtype)
net2:type(dtype)
net1:evaluate()
net2:evaluate()

local x = torch.randn(10, 3, 224, 224):cuda()
local y1 = net1:forward(x)
local y2 = net2:forward(x)
assert(0 == torch.abs(y1 - y2):sum(), 'Outputs do not match')

local dy = torch.randn(#y1):cuda()
local dx1 = net1:updateGradInput(x, dy)
local dx2 = net2:updateGradInput(x, dy)
assert(0 == torch.abs(y1 - y2):sum(), 'gradInput does not match')

utils.restore_gradients(net2)
local grad_w1 = net1:get(1).gradWeight
local grad_w2 = net2:get(1).gradWeight
net1:backward(x, dy)
net2:backward(x, dy)
assert(1e-9 > torch.abs(grad_w1 - grad_w2):mean(), 'gradWeight does not match')
