-- StepConvLSTM_test.lua

dofile '../StepConvLSTM.lua'
inputSize = 3
outputSize = 4
bufferStep = 5
kernelSizeIn = 3
kernelSizeMem = 3
stride = 1
batchSize = 8 
height = 10
width = 10
defaultType = 'nn.DoubleTensor'


net = nn.StepConvLSTM(inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width, defaultType)

input = torch.randn(bufferStep * batchSize, inputSize, height, width)

net:forward(input)

gradOutput = torch.randn(bufferStep * batchSize, outputSize, height, width)

gradInput = net:backward(input, gradOutput)

gradOutput_sub = torch.randn((bufferStep-2) * batchSize, outputSize, height, width)
net:maxBackWard(input, gradOutput_sub)