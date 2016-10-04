require 'torch'
require 'nn'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'


--[[
Prints the options that were used to train a a feedforward model.
--]]


local cmd = torch.CmdLine()
cmd:option('-model', 'models/instance_norm/candy.t7')
local opt = cmd:parse(arg)

print('Loading model from ' .. opt.model)
local checkpoint = torch.load(opt.model)

for k, v in pairs(checkpoint.opt) do
  if type(v) == 'table' then
    v = table.concat(v, ',')
  end
  print(string.format('%s: %s', k, v))
end

