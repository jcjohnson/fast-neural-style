require 'torch'
require 'nn'

local Gram, parent = torch.class('nn.GramMatrix', 'nn.Module')


--[[
A layer to compute the Gram Matrix of inputs.

Input:
- features: A tensor of shape (N, C, H, W) or (C, H, W) giving features for
  either a single image or a minibatch of images.

Output:
- gram: A tensor of shape (N, C, C) or (C, C) giving Gram matrix for input.
--]]


function Gram:__init(normalize)
  parent.__init(self)
  self.normalize = normalize or true
  self.buffer = torch.Tensor()
end


function Gram:updateOutput(input)
  local C, H, W
  if input:dim() == 3 then
    C, H, W = input:size(1), input:size(2), input:size(3)
    local x_flat = input:view(C, H * W)
    self.output:resize(C, C)
    self.output:mm(x_flat, x_flat:t())
  elseif input:dim() == 4 then
    local N = input:size(1)
    C, H, W = input:size(2), input:size(3), input:size(4)
    self.output:resize(N, C, C)
    local x_flat = input:view(N, C, H * W)
    self.output:resize(N, C, C)
    self.output:bmm(x_flat, x_flat:transpose(2, 3))
  end
  if self.normalize then
    -- print('in gram forward; dividing by ', C * H * W)
    self.output:div(C * H * W)
  end
  return self.output
end


function Gram:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W
  if input:dim() == 3 then
    C, H, W = input:size(1), input:size(2), input:size(3)
    local x_flat = input:view(C, H * W)
    self.buffer:resize(C, H * W)
    self.buffer:mm(gradOutput, x_flat)
    self.buffer:addmm(gradOutput:t(), x_flat)
    self.gradInput = self.buffer:view(C, H, W)
  elseif input:dim() == 4 then
    local N = input:size(1)
    C, H, W = input:size(2), input:size(3), input:size(4)
    local x_flat = input:view(N, C, H * W)
    self.buffer:resize(N, C, H * W)
    self.buffer:bmm(gradOutput, x_flat)
    self.buffer:baddbmm(gradOutput:transpose(2, 3), x_flat)
    self.gradInput = self.buffer:view(N, C, H, W)
  end
  if self.normalize then
    self.buffer:div(C * H * W)
  end
  assert(self.gradInput:isContiguous())
  return self.gradInput
end

