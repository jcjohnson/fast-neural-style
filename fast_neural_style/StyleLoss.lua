require 'torch'
require 'nn'

require 'fast_neural_style.GramMatrix'

local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')


function StyleLoss:__init(strength, loss_type, agg_type)
  parent.__init(self)
  self.agg_type = agg_type or 'gram'
  self.strength = strength or 1.0
  self.loss = 0
  self.target = torch.Tensor()

  if self.agg_type == 'gram' then
    self.agg = nn.GramMatrix()
  elseif self.agg_type == 'mean' then
    self.agg = nn.Sequential()
    self.agg:add(nn.Mean(3))
    self.agg:add(nn.Mean(3))
  else
    error('Unsupported aggregation type ' .. agg_type)
  end
  self.agg_out = nil
  
  self.mode = 'none'
  loss_type = loss_type or 'L2'
  if loss_type == 'L2' then
    self.crit = nn.MSECriterion()
  elseif self.crit == 'SmoothL1' then
    self.crit = nn.SmoothL1Criterion()
  else
    error(string.format('invalid loss type "%s"', loss_type))
  end
end


function StyleLoss:updateOutput(input)
  self.agg_out = self.agg:forward(input)
  if self.mode == 'capture' then
    self.target:resizeAs(self.agg_out):copy(self.agg_out)
  elseif self.mode == 'loss' then
    local target = self.target
    if self.agg_out:size(1) > 1 and self.target:size(1) == 1 then
      -- Handle minibatch inputs
      target = target:expandAs(self.agg_out)
    end
    self.loss = self.strength * self.crit(self.agg_out, target)
    self._target = target
  end
  self.output = input
  return self.output
end


function StyleLoss:updateGradInput(input, gradOutput)
  if self.mode == 'capture' or self.mode == 'none' then
    self.gradInput = gradOutput
  elseif self.mode == 'loss' then
    self.crit:backward(self.agg_out, self._target)
    self.crit.gradInput:mul(self.strength)
    self.agg:backward(input, self.crit.gradInput)
    self.gradInput:add(self.agg.gradInput, gradOutput)
  end
  return self.gradInput
end


function StyleLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
