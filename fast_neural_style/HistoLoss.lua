require 'torch'
require 'nn'

local threads = require 'threads'

local function makeCdfInv(img, bins)
  -- things we'll need later.
  local imgmin = img:min()
  local imgmax = img:max()-img:min()
  local cdfinv = torch.zeros(bins)
  local cdfinvcount = torch.ones(bins)
  local cdfsum = 0

  local imgview = img:view(-1)
  
  -- calculate histogram
  local hist = torch.histc(img, bins)
  
  -- calculate probability density function...
  local pmf = hist:div(img:nElement())
  
  -- ... and use that to generate a cumulative density function.
  local cdf = pmf:apply(function(x)
            cdfsum = cdfsum + x
            return cdfsum
            end
           )
		   
  -- we then scale and floor the CDF for generating the inverse CDF
  cdf:mul(bins-1):floor()
  
  -- and then generate the inverse cdf.
  imgview:apply(function(x)
                local y = math.floor(((x-imgmin)/(imgmax+1e-11))*(bins-1)+1)
				y = cdf[y]+1
				cdfinv[y] = cdfinv[y] + x
				cdfinvcount[y] = cdfinvcount[y] + 1
                end
			   )
  cdfinv:cdiv(cdfinvcount)
  
  -- to improve results, replace all unfilled inverse CDF bins with linear interpolated values.
  cdfinv[bins] = cdfinv:max()
  if math.ceil(cdfinv:max()) ~= 0 then
    for i = 2, cdfinv:size()[1] do
      local count = 1
      local temp1 = temp1 or cdfinv[i-1]
      local temp2 = 0
      if cdfinv[i] == 0 then
        while cdfinv[i-1+count] == 0 do
          count = count + 1
          temp2 = cdfinv[i-1+count]
        end
        if count < 2 then
        end
        cdfinv[i] = temp1*(1/count)+temp2*(1-(1/count))
      else
        temp1 = cdfinv[i]
      end
    end
  end
  
  return cdfinv
end

local function histoMatch(img, cdfinv, bins)
   -- things we'll need later.
  local imgmin = img:min()
  local imgmax = img:max()-img:min()
  local cdfsum = 0

  local imgview = img:view(-1)
  
  -- calculate histogram
  local hist = torch.histc(img, bins)
  
  -- calculate probability density function...
  local pmf = hist:div(img:nElement())
  
  -- ... and use that to generate a cumulative density function.
  local cdf = pmf:apply(function(x)
            cdfsum = cdfsum + x
            return cdfsum
            end
           )
  -- finally, we use the generated CDF to match the histograms.
  local function invert(img)
    img = math.floor(((img-imgmin)/(imgmax+1e-11))*(bins-1)+1)
	img = math.floor(cdf[img]*(bins-1)+1)
	return cdfinv[img]
  end
  imgview:apply(invert)
  
  return(img)
end

local HistoLoss, parent = torch.class('nn.HistoLoss', 'nn.Module')

function HistoLoss:__init(strength, bins, n_threads)
  parent.__init(self)
  self.strength = strength
  self.target = nil
  self.loss = 0
  self.bins = bins
  self.mode = 'none'
  self.H = nil
  self.crit = nn.MSECriterion()
  self.crit.sizeAverage = true
  self.threads = n_threads or 6
end

function HistoLoss:updateOutput(input)
  -- since creating an opencl/CUDA kernel for this is non-trivial,
  -- instead i've chosen to thread the fuck out of it.
  local pool = threads.Threads(self.threads)
  local bins_thread = self.bins
  if self.mode == 'capture' then
    self.target = torch.Tensor(input:size()[1], input:size()[2], self.bins)
    for i = 1, input:size()[1] do
      for j = 1, input:size()[2] do
        local input_thread = input[i][j]:clone()
        local target_thread = self.target[i][j]:clone()
        pool:addjob(
          function()
          target_thread = makeCdfInv(input_thread, bins_thread)
          return target_thread
          end,
      
          function(target_thread)
          self.target[i][j] = target_thread
          end
        )
      end
    end
    pool:synchronize()
    pool:terminate()
  elseif self.mode == 'loss' then
    self.H = input:clone()
    for i = 1, input:size()[1] do
      for j = 1, input:size()[2] do
        local target_thread = self.target[1][j]:clone()
        local input_thread = input[i][j]:clone()
        local H_thread = self.H[i][j]:clone()
        pool:addjob(
        function()
          H_thread = histoMatch(input_thread, target_thread, bins_thread)
          return H_thread
        end,
        function(H_thread)
          self.H[i][j] = H_thread
        end
        )
      end
    end
    pool:synchronize()
    pool:terminate()
    self.loss = self.crit:forward(input, self.H)
    self.loss = self.loss * self.strength
  end
  self.output = input
  return self.output
end

function HistoLoss:updateGradInput(input, gradOutput)
  if self.mode == 'capture' or self.mode == 'none' then
    self.gradInput = gradOutput
  elseif self.mode == 'loss' then
    self.gradInput = self.crit:backward(input, self.H)
    self.gradInput:mul(self.strength)
    self.gradInput:add(gradOutput)
  end
  return self.gradInput
end

function HistoLoss:setMode(mode)
  if mode ~= 'capture' and mode ~= 'loss' and mode ~= 'none' then
    error(string.format('Invalid mode "%s"', mode))
  end
  self.mode = mode
end
