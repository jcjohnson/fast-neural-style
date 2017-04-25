require 'torch'
require 'nn'
require 'fast_neural_style.InstanceNormalization'
require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'

-- Prints the command-line arguments used to train a particular model


local cmd = torch.CmdLine()
cmd:option('-model', 'models/instance_norm/candy.t7')

function main()
  local opt = cmd:parse(arg)
  local checkpoint = torch.load(opt.model)
  for k, v in pairs(checkpoint.opt) do
    if type(v) == 'table' then
      v = table.concat(v, ',')
    end
    print('-' .. k, v)
  end
end


main()

