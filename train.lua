require 'torch'
require 'optim'
require 'image'

require 'fast_neural_style.DataLoader'
require 'fast_neural_style.PerceptualCriterion'

local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local models = require 'fast_neural_style.models'

local cmd = torch.CmdLine()


--[[
Train a feedforward style transfer model
--]]

-- Generic options
cmd:option('-arch', 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3')
cmd:option('-use_instance_norm', 1)
cmd:option('-task', 'style', 'style|upsample')
cmd:option('-h5_file', 'data/ms-coco-256.h5')
cmd:option('-padding_type', 'reflect-start')
cmd:option('-tanh_constant', 150)
cmd:option('-preprocessing', 'vgg')
cmd:option('-resume_from_checkpoint', '')

-- Generic loss function options
cmd:option('-pixel_loss_type', 'L2', 'L2|L1|SmoothL1')
cmd:option('-pixel_loss_weight', 0.0)
cmd:option('-percep_loss_weight', 1.0)
cmd:option('-tv_strength', 1e-6)

-- Options for feature reconstruction loss
cmd:option('-content_weights', '1.0')
cmd:option('-content_layers', '16')
cmd:option('-loss_network', 'models/vgg16.t7')

-- Options for style reconstruction loss
cmd:option('-style_image', 'images/styles/candy.jpg')
cmd:option('-style_image_size', 256)
cmd:option('-style_weights', '5.0')
cmd:option('-style_layers', '4,9,16,23')
cmd:option('-style_target_type', 'gram', 'gram|mean')

-- Upsampling options
cmd:option('-upsample_factor', 4)

-- Optimization
cmd:option('-num_iterations', 40000)
cmd:option('-max_train', -1)
cmd:option('-batch_size', 4)
cmd:option('-learning_rate', 1e-3)
cmd:option('-lr_decay_every', -1)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-weight_decay', 0)

-- Checkpointing
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-checkpoint_every', 1000)
cmd:option('-num_val_batches', 10)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')


 function main()
  local opt = cmd:parse(arg)

  -- Parse layer strings and weights
  opt.content_layers, opt.content_weights =
    utils.parse_layers(opt.content_layers, opt.content_weights)
  opt.style_layers, opt.style_weights =
    utils.parse_layers(opt.style_layers, opt.style_weights)

  -- Figure out preprocessing
  if not preprocess[opt.preprocessing] then
    local msg = 'invalid -preprocessing "%s"; must be "vgg" or "resnet"'
    error(string.format(msg, opt.preprocessing))
  end
  preprocess = preprocess[opt.preprocessing]

  -- Figure out the backend
  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

  -- Build the model
  local model = nil
  if opt.resume_from_checkpoint ~= '' then
    print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
    model = torch.load(opt.resume_from_checkpoint).model:type(dtype)
  else
    print('Initializing model from scratch')
    model = models.build_model(opt):type(dtype)
  end
  if use_cudnn then cudnn.convert(model, cudnn) end
  model:training()
  print(model)
  
  -- Set up the pixel loss function
  local pixel_crit
  if opt.pixel_loss_weight > 0 then
    if opt.pixel_loss_type == 'L2' then
      pixel_crit = nn.MSECriterion():type(dtype)
    elseif opt.pixel_loss_type == 'L1' then
      pixel_crit = nn.AbsCriterion():type(dtype)
    elseif opt.pixel_loss_type == 'SmoothL1' then
      pixel_crit = nn.SmoothL1Criterion():type(dtype)
    end
  end

  -- Set up the perceptual loss function
  local percep_crit
  if opt.percep_loss_weight > 0 then
    local loss_net = torch.load(opt.loss_network)
    local crit_args = {
      cnn = loss_net,
      style_layers = opt.style_layers,
      style_weights = opt.style_weights,
      content_layers = opt.content_layers,
      content_weights = opt.content_weights,
      agg_type = opt.style_target_type,
    }
    percep_crit = nn.PerceptualCriterion(crit_args):type(dtype)

    if opt.task == 'style' then
      -- Load the style image and set it
      local style_image = image.load(opt.style_image, 3, 'float')
      style_image = image.scale(style_image, opt.style_image_size)
      local H, W = style_image:size(2), style_image:size(3)
      style_image = preprocess.preprocess(style_image:view(1, 3, H, W))
      percep_crit:setStyleTarget(style_image:type(dtype))
    end
  end

  local loader = DataLoader(opt)
  local params, grad_params = model:getParameters()


  local function shave_y(x, y, out)
    if opt.padding_type == 'none' then
      local H, W = x:size(3), x:size(4)
      local HH, WW = out:size(3), out:size(4)
      local xs = (H - HH) / 2
      local ys = (W - WW) / 2
      return y[{{}, {}, {xs + 1, H - xs}, {ys + 1, W - ys}}]
    else
      return y
    end
  end
  

  local function f(x)
    assert(x == params)
    grad_params:zero()
    
    local x, y = loader:getBatch('train')
    x, y = x:type(dtype), y:type(dtype)

    -- Run model forward
    local out = model:forward(x)
    local grad_out = nil

    -- This is a bit of a hack: if we are using reflect-start padding and the
    -- output is not the same size as the input, lazily add reflection padding
    -- to the start of the model so the input and output have the same size.
    if opt.padding_type == 'reflect-start' and x:size(3) ~= out:size(3) then
      local ph = (x:size(3) - out:size(3)) / 2
      local pw = (x:size(4) - out:size(4)) / 2
      local pad_mod = nn.SpatialReflectionPadding(pw, pw, ph, ph):type(dtype)
      model:insert(pad_mod, 1)
      out = model:forward(x)
    end

    y = shave_y(x, y, out)

    -- Compute pixel loss and gradient
    local pixel_loss = 0
      if pixel_crit then
      pixel_loss = pixel_crit:forward(out, y)
      pixel_loss = pixel_loss * opt.pixel_loss_weight
      local grad_out_pix = pixel_crit:backward(out, y)
      if grad_out then
        grad_out:add(opt.pixel_loss_weight, grad_out_pix)
      else
        grad_out_pix:mul(opt.pixel_loss_weight)
        grad_out = grad_out_pix
      end
    end

    -- Compute perceptual loss and gradient
    local percep_loss = 0
    if percep_crit then
      local target = {content_target=y}
      percep_loss = percep_crit:forward(out, target)
      percep_loss = percep_loss * opt.percep_loss_weight
      local grad_out_percep = percep_crit:backward(out, target)
      if grad_out then
        grad_out:add(opt.percep_loss_weight, grad_out_percep)
      else
        grad_out_percep:mul(opt.percep_loss_weight)
        grad_out = grad_out_percep
      end
    end

    local loss = pixel_loss + percep_loss

    -- Run model backward
    model:backward(x, grad_out)

    -- Add regularization
    -- grad_params:add(opt.weight_decay, params)
 
    return loss, grad_params
  end


  local optim_state = {learningRate=opt.learning_rate}
  local train_loss_history = {}
  local val_loss_history = {}
  local val_loss_history_ts = {}
  local style_loss_history = nil
  if opt.task == 'style' then
    style_loss_history = {}
    for i, k in ipairs(opt.style_layers) do
      style_loss_history[string.format('style-%d', k)] = {}
    end
    for i, k in ipairs(opt.content_layers) do
      style_loss_history[string.format('content-%d', k)] = {}
    end
  end

  local style_weight = opt.style_weight
  for t = 1, opt.num_iterations do
    local epoch = t / loader.num_minibatches['train']

    local _, loss = optim.adam(f, params, optim_state)

    table.insert(train_loss_history, loss[1])

    if opt.task == 'style' then
      for i, k in ipairs(opt.style_layers) do
        table.insert(style_loss_history[string.format('style-%d', k)],
          percep_crit.style_losses[i])
      end
      for i, k in ipairs(opt.content_layers) do
        table.insert(style_loss_history[string.format('content-%d', k)],
          percep_crit.content_losses[i])
      end
    end

    print(string.format('Epoch %f, Iteration %d / %d, loss = %f',
          epoch, t, opt.num_iterations, loss[1]), optim_state.learningRate)

    if t % opt.checkpoint_every == 0 then
      -- Check loss on the validation set
      loader:reset('val')
      model:evaluate()
      local val_loss = 0
      print 'Running on validation set ... '
      local val_batches = opt.num_val_batches
      for j = 1, val_batches do
        local x, y = loader:getBatch('val')
        x, y = x:type(dtype), y:type(dtype)
        local out = model:forward(x)
        y = shave_y(x, y, out)
        local pixel_loss = 0
        if pixel_crit then
          pixel_loss = pixel_crit:forward(out, y)
          pixel_loss = opt.pixel_loss_weight * pixel_loss
        end
        local percep_loss = 0
        if percep_crit then
          percep_loss = percep_crit:forward(out, {content_target=y})
          percep_loss = opt.percep_loss_weight * percep_loss
        end
        val_loss = val_loss + pixel_loss + percep_loss
      end
      val_loss = val_loss / val_batches
      print(string.format('val loss = %f', val_loss))
      table.insert(val_loss_history, val_loss)
      table.insert(val_loss_history_ts, t)
      model:training()

      -- Save a JSON checkpoint
      local checkpoint = {
        opt=opt,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_loss_history_ts=val_loss_history_ts,
        style_loss_history=style_loss_history,
      }
      local filename = string.format('%s.json', opt.checkpoint_name)
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)

      -- Save a torch checkpoint; convert the model to float first
      model:clearState()
      if use_cudnn then
        cudnn.convert(model, nn)
      end
      model:float()
      checkpoint.model = model
      filename = string.format('%s.t7', opt.checkpoint_name)
      torch.save(filename, checkpoint)

      -- Convert the model back
      model:type(dtype)
      if use_cudnn then
        cudnn.convert(model, cudnn)
      end
      params, grad_params = model:getParameters()
    end

    if opt.lr_decay_every > 0 and t % opt.lr_decay_every == 0 then
      local new_lr = opt.lr_decay_factor * optim_state.learningRate
      optim_state = {learningRate = new_lr}
    end

  end

end


main()

