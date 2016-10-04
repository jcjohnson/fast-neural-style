require 'torch'
require 'nn'

local layer_utils = require 'neuralstyle2.layer_utils'


local tester = torch.Tester()
local tests = torch.TestSuite()


function tests.layer_string_to_nums_test()
  local test_cases = {
    {"1", {1}},
    {"1-2", {1, 2}},
    {"10-21", {10, 21}},
    {"20-21-31", {20, 21, 31}},
  }
  for _, test_case in ipairs(test_cases) do
    local input = test_case[1]
    local expected_output = test_case[2]
    local output = layer_utils.layer_string_to_nums(input)
    tester:assertTableEq(output, expected_output)
  end
end


function tests.compare_layer_strings_test()
  local test_cases = {
    {"1", "2", true},
    {"1", "1-2", false},
    {"1-2-3", "1-2-4", true},
    {"1-2-4", "1-2-3", false},
    {"10-3-25", "10-2", false},
    {"10-2-23", "10-2-24-2", true},
    {"5-2-3-4", "5-2-4-4", true},
    {"5-2-3-4", "5-2-2-4", false},
  }
  for _, test_case in ipairs(test_cases) do
    local s1, s2 = test_case[1], test_case[2]
    local expected_output = test_case[3]
    local output = layer_utils.compare_layer_strings(s1, s2)
    tester:asserteq(output, expected_output)
  end
end


function tests.get_get_insert_layer()
  local all_tags = {}
  local function tag_module(tag, m)
    table.insert(all_tags, tag)
    m._tag = tag
    return m
  end
  
  --[[
  We are building the following net:

  nn.Sequential {
    (1) nn.Linear(1, 1)
    (2) nn.Sequential() {
      (2-1) nn.Linear(2, 1)
      (2-2) nn.Linear(2, 2)
      (2-3) nn.Sequential() {
        (2-3-1) nn.Linear(2, 2)
      }
    }
    (3) nn.Sequential() {
      (3-1) nn.Linear(3, 1)
      (3-2) nn.Linear(3, 2)
    }
    (4) nn.Linear(4, 4)
  }
  ]]
  local net = nn.Sequential()
  net:add(tag_module('1', nn.Linear(1, 1)))
  net:add(tag_module('2', nn.Sequential()))
  net:get(2):add(tag_module('2-1', nn.Linear(2, 1)))
  net:get(2):add(tag_module('2-2', nn.Linear(2, 2)))
  net:get(2):add(tag_module('2-3', nn.Sequential()))
  net:get(2):get(3):add(tag_module('2-3-1', nn.Linear(2, 2)))
  net:add(tag_module('3', nn.Sequential()))
  net:get(3):add(tag_module('3-1', nn.Linear(3, 1)))
  net:get(3):add(tag_module('3-2', nn.Linear(3, 2)))
  net:add(tag_module('4', nn.Linear(4, 4)))

  local function test_tagged_layers()
    for _, s in ipairs(all_tags) do
      local layer = layer_utils.get_layer(net, s)
      tester:asserteq(layer._tag, s)
    end
  end

  -- Make sure we can get all the original layer properly
  test_tagged_layers()
 
  -- Insert a bunch of new layers at various points in the net
  local layer1 = nn.Linear(50, 50)
  layer_utils.insert_after(net, '1', layer1)
  local layer2 = nn.Linear(60, 60)
  layer_utils.insert_after(net, '2-2', layer2)
  local layer3 = nn.Linear(70, 70)
  layer_utils.insert_after(net, '2-3', layer3)
  local layer4 = nn.Linear(80, 80)
  layer_utils.insert_after(net, '2-3', layer4)
  local layer5 = nn.Linear(90, 90)
  layer_utils.insert_after(net, '3-1', layer5)
  local layer6 = nn.Linear(100, 100)
  layer_utils.insert_after(net, '3-1', layer6)

  --[[
  After all these insertions, the net should look like this:
  nn.Sequential {
    (1) nn.Linear(1, 1)
    (layer1)
    (2) nn.Sequential() {
      (2-1) nn.Linear(2, 1)
      (2-2) nn.Linear(2, 2)
      (layer2)
      (2-3) nn.Sequential() {
        (2-3-1) nn.Linear(2, 2)
      }
      (layer4)
      (layer3)
    }
    (3) nn.Sequential() {
      (3-1) nn.Linear(3, 1)
      (layer6)
      (layer5)
      (3-2) nn.Linear(3, 2)
    }
    (4) nn.Linear(4, 4)
  }
  ]]
  
  -- Make sure the new layers were put in the correct places
  tester:asserteq(layer1, net:get(2))
  tester:asserteq(layer2, net:get(3):get(3))
  tester:asserteq(layer3, net:get(3):get(6))
  tester:asserteq(layer4, net:get(3):get(5))
  tester:asserteq(layer5, net:get(4):get(3))
  tester:asserteq(layer6, net:get(4):get(2))

  -- Also make sure we can still get all the old layers
  test_tagged_layers()
end


tester:add(tests)
tester:run()
