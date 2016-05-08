print(os.date("today is %c"))
------------------------------------------------------------------------
--[[ 
`RecursiveSequential`

Encapsulates a Module. 
input: (batchSize, inputSize, height, width)
self.output: (bufferSize, batchSize, outputSize, height, width)
notice that self.output = [currentOutput1, currentOutput2, currentOutput3...]
for each step the input concated to be: [input, currentOutput1, currentOutput2, ...]

for t = 1, self.bufferSize
   Input is a one bacth tensor in dim = 4: 
      currentInput = (t == 1)? input : self.output[t - 1]

   1. forward
      call updateOutput(currentInput) sequentially, pass through all #modules

      get currentOutput: (batchSize, outputSize, height, width)
      store output in self.output[t]

   2. calculate loss:
      fetch the target slice: 
         currentTarget = target[t]
      currentError = criterion:forward(currentOutput, currentTarget)
      currentGradOutput = criterion:backward(currentOutput, currentTarget)

   3. backward and getGradInput:

         call updateGradInput(currentGradOutput) sequentially
         call accGradParameters(currentInput, currentGradOutput) sequentially

         get currentGradInput, 
         accumulate it to self.GradInput

end

]]--
------------------------------------------------------------------------
local RecursiveSequential, parent = torch.class('nn.RecursiveSequential', 'nn.Sequential')

function RecursiveSequential:__init(bufferSize)
   parent.__init(self)
   assert(bufferSize, 'for RecursiveSequential bufferSize is required')
   self.bufferSize = bufferSize
   self.output = torch.Tensor() -- wait for resize

   self.flowOutput = {} -- serve as a buffer
   -- table of buffers used for evaluation
   
   -- so that these buffers aren't serialized :
   
   -- default is to forget previous inputs before each forward()
   
end

function RecursiveSequential:autoForwardAndBackward(input, target, criterion)
   assert(torch.isTensor(inputBatch))
   assert(input:dim() == 4, 'fail input:dim() == 4')
   assert(target:dim() == 5, 'fail: target:dim() == 5')
   assert(target:size(1) == self.bufferSize, 'target dimension(1)'..target:size(1)..' mismatch bufferSize'..self.bufferSize)



function RecursiveSequential:autoForwardAndBackward(input, outputSeqLength)
   local outputSequenceLength = outputSeqLength or self.bufferSize
   assert(input:dim() == 4) 






