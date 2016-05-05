require 'nn'
require 'rnn'
require 'FastConvLSTM'
require 'StepConvLSTM'
local rnntest = {}
local mytester = torch.Tester()

function rnntest.lstm()
-- parameters

local lstmSize_input = 3
local lstmSize_output = 7
local batchSize = 2
local nStep = 4


inputSize = lstmSize_input
outputSize = lstmSize_output
bufferStep = nStep
kernelSizeIn = 3
kernelSizeMem = 3
stride = 1

height = 50
width = 50

-- 
nn.StepConvLSTM.usenngraph = true
print(inputSize, outputSize,bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width)
lstm3 = nn.StepConvLSTM(inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width)
local param3, gradParams = lstm3:getParameters()
param3:fill(1)
temp = torch.Tensor():resizeAs(param3):fill(1)
print("size of parameters of lstm3", param3:size())
lstm3:__tostring__()

nn.StepConvLSTM.usenngraph = false
print(inputSize, outputSize,bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width)
lstm4 = nn.StepConvLSTM(inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width)
local param4, gradParams = lstm4:getParameters()
-- param4 = param3
local param5, gradParams = lstm4.modules[1].modules[1].modules[1].modules[2]:getParameters()
print('=========	')
lstm2 = nn.FastConvLSTM(lstmSize_input, lstmSize_output, nStep, 3, 3, 1, batchSize)  -- with nngraph
local param2, gradParams2 = lstm2.modules[1].modules[1].modules[1].modules[2]:getParameters()
param2:set(param5)

print("size of parameters of lstm2:",param2:size())
print(lstm2.modules[1].modules[1].modules[1].modules[2])

-- prepare input:
input = torch.Tensor(nStep, batchSize, inputSize, height, width)
gradOutput = torch.Tensor(nStep, batchSize, outputSize, height, width)
inputTable = {}
gradOutputTable = {}

local H = height
local W = width
for step=1, nStep do
  input[step] = torch.randn(batchSize, lstmSize_input, H, W) * 10
  gradOutput[step] = torch.randn(batchSize, lstmSize_output, H, W)
  inputTable[step] = torch.randn(batchSize, lstmSize_input, H, W)
  gradOutputTable[step] = torch.randn(batchSize, lstmSize_output, H, W)
  
end
assert(input:dim() == 5)
-- test forward

output3 = lstm3:forward(input)
output3 = lstm3:forward(input)
-- print(input[1])
output4 = lstm4:forward(input)
-- print(output3[1])
-- print(lstm4.modules[1].output[1])
-- mytester:assertTensorEq(output3, output4, 0.00000001)

output2 = lstm2:forward(input[1])
gradOutput = torch.Tensor():resizeAs(output3):normal(10,0.1)
lstm3:backward(input, gradOutput)
param3, gradParams = lstm3.modules[1]:getParameters()
mytester:assertTensorNe(param3, temp, 0.00000001)
-- sprint(gradParams)
-- print(output2:size())
-- print(output3)
-- mytester:assertTensorEq(output3[1], output2, 0.00000001)

end

mytester = torch.Tester()
mytester:add(rnntest)
-- math.randomseed(os.time())
mytester:run()

