require 'nn'
dofile '../RecursiveSequential.lua'
dofile '../StepConvLSTM.lua'

inputSize = 4
outputSize = 5
batchSize = 3
height = 10
width = 10
bufferSize = 6
bufferStepLSTM = 1
encoder = nn.SpatialConvolution(inputSize, inputSize*2, 3, 3, 1, 1, 1, 1)

net = nn.StepConvLSTM(inputSize*2, outputSize, bufferStepLSTM, 3, 3, 1, batchSize, height, width, "nn.DoubleTensor")

decoder = nn.SpatialConvolution(outputSize, inputSize, 3, 3, 1, 1, 1, 1)
mlp = nn.RecursiveSequential(bufferSize):add(encoder):add(net):add(decoder)

target = torch.Tensor(bufferSize, batchSize, inputSize, height, width):normal(3, 0.1)
input = torch.Tensor(batchSize, inputSize, height, width):normal(2, 0.1)

output,accErr,gradInput = mlp:autoForwardAndBackward(input, target)



-- output = t

-- gradInput = t[2]

--accErr = t[3]

print("outputsize: \n", output:size())

print("gradInput size \n", gradInput:size())