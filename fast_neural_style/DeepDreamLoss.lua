require 'torch'
require 'nn'

local layer, parent = torch.class('nn.DeepDreamLoss', 'nn.Module')


function layer:__init(strength, max_grad)
  parent.__init(self)
  self.strength = strength or 1e-5
  self.max_grad = max_grad or 100.0
  self.clipped = torch.Tensor()
  self.loss = 0
end


function layer:updateOutput(input)
  self.output = input
  return self.output
end


function layer:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.clipped:resizeAs(input):clamp(input, -self.max_grad, self.max_grad)
  self.gradInput:add(-self.strength, self.clipped)
  return self.gradInput
end

