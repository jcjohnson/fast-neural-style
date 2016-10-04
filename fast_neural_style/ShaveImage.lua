local layer, parent = torch.class('nn.ShaveImage', 'nn.Module')

function layer:__init(size)
  parent.__init(self)
  self.size = size
end


function layer:updateOutput(input)
  local N, C = input:size(1), input:size(2)
  local H, W = input:size(3), input:size(4)
  local s = self.size
  self.output:resize(N, C, H - 2 * s, W - 2 * s)
  self.output:copy(input[{{}, {}, {s + 1, H - s}, {s + 1, W - s}}])
  return self.output
end


function layer:updateGradInput(input, gradOutput)
  local N, C = input:size(1), input:size(2)
  local H, W = input:size(3), input:size(4)
  local s = self.size
  self.gradInput:resizeAs(input):zero()
  self.gradInput[{{}, {}, {s + 1, H - s}, {s + 1, W - s}}]:copy(gradOutput)
  return self.gradInput
end

