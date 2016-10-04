require 'torch'
require 'nn'
local cjson = require 'cjson'


local M = {}


-- Parse a string of comma-separated numbers
-- For example convert "1.0,3.14" to {1.0, 3.14}
function M.parse_num_list(s)
  local nums = {}
  for _, ss in ipairs(s:split(',')) do
    table.insert(nums, tonumber(ss))
  end
  return nums
end


-- Parse a layer string and associated weights string.
-- The layers string is a string of comma-separated layer strings, and the
-- weight string contains comma-separated numbers. If the weights string
-- contains only a single number it is duplicated to be the same length as the
-- layers.
function M.parse_layers(layers_string, weights_string)
  local layers = layers_string:split(',')
  local weights = M.parse_num_list(weights_string)
  if #weights == 1 and #layers > 1 then
    -- Duplicate the same weight for all layers
    local w = weights[1]
    weights = {}
    for i = 1, #layers do
      table.insert(weights, w)
    end
  elseif #weights ~= #layers then
    local msg = 'size mismatch between layers "%s" and weights "%s"'
    error(string.format(msg, layers_string, weights_string))
  end
  return layers, weights
end


function M.setup_gpu(gpu, backend, use_cudnn)
  local dtype = 'torch.FloatTensor'
  if gpu >= 0 then
    if backend == 'cuda' then
      require 'cutorch'
      require 'cunn'
      cutorch.setDevice(gpu + 1)
      dtype = 'torch.CudaTensor'
      if use_cudnn then
        require 'cudnn'
        cudnn.benchmark = true
      end
    elseif backend == 'opencl' then
      require 'cltorch'
      require 'clnn'
      cltorch.setDevice(gpu + 1)
      dtype = torch.Tensor():cl():type()
      use_cudnn = false
    end
  else
    use_cudnn = false
  end
  return dtype, use_cudnn
end


function M.clear_gradients(m)
  if torch.isTypeOf(m, nn.Container) then
    m:applyToModules(M.clear_gradients)
  end
  if m.weight and m.gradWeight then
    m.gradWeight = m.gradWeight.new()
  end
  if m.bias and m.gradBias then
    m.gradBias = m.gradBias.new()
  end
end


function M.restore_gradients(m)
  if torch.isTypeOf(m, nn.Container) then
    m:applyToModules(M.restore_gradients)
  end
  if m.weight and m.gradWeight then
    m.gradWeight = m.gradWeight.new(#m.weight):zero()
  end
  if m.bias and m.gradBias then
    m.gradBias = m.gradBias.new(#m.bias):zero()
  end
end


function M.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end


function M.write_json(path, j)
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

local IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'ppm', 'pgm'}
function M.is_image_file(filename)
  -- Hidden file are not images
  if string.sub(filename, 1, 1) == '.' then
    return false
  end
  -- Check against a list of known image extensions
  local ext = string.lower(paths.extname(filename))
  for _, image_ext in ipairs(IMAGE_EXTS) do
    if ext == image_ext then
      return true
    end
  end
  return false
end


function M.median_filter(img, r)
  local u = img:unfold(2, r, 1):contiguous()
  u = u:unfold(3, r, 1):contiguous()
  local HH, WW = u:size(2), u:size(3)
  local dtype = u:type()
  -- Median is not defined for CudaTensors, cast to float and back
  local med = u:view(3, HH, WW, r * r):float():median():type(dtype)
  return med[{{}, {}, {}, 1}]
end


return M

