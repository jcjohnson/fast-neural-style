require 'torch'
require 'nn'

local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')


--[[
Module to compute content loss in-place.

The module can be in one of three modes: "none", "capture", or "loss", which
behave as follows:
- "none": This module does nothing; it is basically nn.Identity().
- "capture": On the forward pass, inputs are captured as targets; otherwise it
  is the same as an nn.Identity().
- "loss": On the forward pass, compute the distance between input and
  self.target, store the result in self.loss, and return input. On the backward
  pass, add compute the gradient of self.loss with respect to the inputs, and
  add this value to the upstream gradOutput to produce gradInput.
--]]

function ContentLoss:__init(strength, loss_type)
  parent.__init(self)
  self.strength = strength or 1.0
  self.loss = 0
  self.target = torch.Tensor()

  self.mode = 'none'
  loss_type = loss_type or 'L2'
  if loss_type == 'L2' then
    self.crit = nn.MSECriterion()
  elseif loss_type == 'SmoothL1' then
    self.crit = nn.SmoothL1Criterion()
  else
    error(string.format('Invalid loss_type "%s"', loss_type))
  end
end


function ContentLoss:updateOutput(input)
  if self.mode == 'capture' then
    self.target:resizeAs(input):copy(input)
  elseif self.mode == 'loss' then
    self.loss = self.strength * self.crit:forward(input, self.target)
  end
  self.output = input
  return self.output
end


function ContentLoss:updateGradInput(input, gradOutput)
  if self.mode == 'capture' or self.mode == 'none' then
    self.gradInput = gradOutput
  elseif self.mode == 'loss' then
    self.gradInput = self.crit:backward(input, self.target)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  end
  return self.gradInput
end


function ContentLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
