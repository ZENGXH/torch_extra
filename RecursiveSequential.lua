print(os.date("today is %c"))
------------------------------------------------------------------------
--[[ 
`RecursiveSequential`

Encapsulates a Module. 
input: (batchSize, inputSize, height, width)
self.output: (bufferSize, batchSize, outputSize, height, width)

notice that self.output = [currentOutput1, currentOutput2, currentOutput3...]
for each step the input concated to be: [input, currentOutput1, currentOutput2, ...]

for idBuffer = 1, self.bufferSize
   Input is a one bacth tensor in dim = 4: 
      currentInput = (t == 1)? input : self.output[t - 1]

   1. forward
      call updateOutput(currentInput) sequentially, pass through all #modules
      **********************************************************
      * currentOutput = input                                  *
      * currentOutput = recursiveUpdateOutput(currentOutput)   *
      **********************************************************
      get self.output[idBuffer] = currentOutput: (batchSize, outputSize, height, width)
      store output in self.output[idBuffer]

   2. calculate loss:
      fetch the target slice: 
         currentTarget = target[t]
      currentError = criterion:forward(currentOutput, currentTarget)
      currentGradOutput = criterion:backward(currentOutput, currentTarget)
      

   3. backward and getGradInput:

      call updateGradInput(currentGradOutput) sequentially
      call accGradParameters(currentInput, currentGradOutput) sequentially
      *********************************************************************
      * currentGradInput = currentGradOutput                              *
      * currentGradInput = recursiveBackward({input, currentGradInput})   *
      *********************************************************************
         get currentGradInput, 
      accumulate it to self.GradInput

end

]]--
------------------------------------------------------------------------
local RecursiveSequential, parent = torch.class('nn.RecursiveSequential', 'nn.Container')

function RecursiveSequential:__init(bufferSize)
   parent.__init(self)
   assert(bufferSize, 'for RecursiveSequential bufferSize is required')
   self.bufferSize = bufferSize
   self.output = torch.Tensor() -- wait for resize
   self:start()
   self.flowOutput = {} -- serve as a buffer
   self.criterion = nn.MSECriterion()

   -- table of buffers used for evaluation
   
   -- so that these buffers aren't serialized :
   
   -- default is to forget previous inputs before each forward()
   -- self.setCriterion()
   
end


function RecursiveSequential:getDimension(x, outputSize)
   local batchSize = x:size(1)
   local height = x:size(3)
   local width = x:size(4)
   self.output:resize(self.bufferSize, batchSize, outputSize, height, width)
   self.gradOutput:resize(batchSize, outputSize, height, width)
   self.gradInput:resize(batchSize, x:size(2), height, width)
end

function RecursiveSequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   -- self.output = module.output
   return self
end

function RecursiveSequential:setCriterion(criterion)
   local criterion = criterion or nn.MSECriterion()
      self.criterion = nn.MSECriterion():float()

   return
end



function RecursiveSequential:autoForwardAndBackward(input, target)
   self.step = 1
   -- input size: batch, inputSize, height, width
   -- output size: bufferSize, batchSize, outputSize, height, width
   -- target size: bufferSize, batchSize, outputSize, height, width
   -- gradOutput size: batchSize, outputSize, height, width
   -- gradInput size: batchSize, inputSize, height, width

   assert(self.criterion, 'criterion required')
   -- print(criterion)
   -- assert(torch.isTypeOf(criterion, 'nn.Criterion'), 'criterion required')
   assert(torch.isTensor(input))
   assert(input:dim() == 4, 'fail input:dim() == 4')
   assert(target:dim() == 5, 'fail: target:dim() == 5')
   assert(target:size(1) == self.bufferSize, 'target dimension(1)'..target:size(1)..' mismatch bufferSize'..self.bufferSize)
   assert(target:size(2) == input:size(1), 'batchSize unmatch:'..target:size(2)..input:size(1) ) -- batchSize
   assert(target:size(4) == input:size(3)) -- height
   assert(target:size(5) == input:size(4)) -- width
   self:getDimension(input, target:size(3))
--[[
   typeChecking(input, self.runtimeType)
   typeChecking(self.output, self.runtimeType)
   typeChecking(self.gradOutput, self.runtimeType)
   typeChecking(self.criterion, self.runtimeType)
]]--
   assert(self.output:size(1) == self.bufferSize)
   assert(self.output:dim() == 5)
   local accErr = 0

   local currentOutput
   -- 
   for idBufferStep = 1, self.bufferSize do
      ---- [[ start forward ]] ----
      print('idBufferStep#', idBufferStep)
      
      
      if idBufferStep == 1 then currentOutput = input else currentOutput = self.output[idBufferStep - 1] end
      assert(currentOutput, 'lack currentOutput')
      -- in current idBufferStep, run all the module sequentially:

      -- **********************************************
      for idModules = 1, #self.modules do
         print('forward modules', idModules,self.modules[idModules])
         print('input: ', currentOutput:size())
         currentOutput = self:rethrowErrors(self.modules[idModules], idModules, 'updateOutput', currentOutput)
      end -- end forward for all modules
      -- **********************************************

      self.output[idBufferStep] = currentOutput
      ---- [[ end forward ]] ----

      ---- [[ start backward ]] ---- 

      -- get error and gradOutput
      

      -- accumulating all the errors for all step:
      accErr = accErr + self.criterion:forward(self.output[idBufferStep], target[idBufferStep])

      local currentGradOutput = self.criterion:backward(self.output[idBufferStep], target[idBufferStep])
      assert(currentGradOutput, 'currentGradOutput empty')

      -- get gradOutput and accGradParameters
      -- local currentGradInput = currentGradOutput
      for idModules = #self.modules,1, -1 do
         -- print(self.modules[idModules])
         -- local parameters, gradParameters = self.modules[idModules]:getParameters()
         -- parameters = parameters:float()
         -- gradParameters = gradParameters:float()
         if idModules == 1 then 
            if idBufferStep == 1 then
               currentInput = input 
            else
               currentInput = self.output[idBufferStep - 1]
            end
         else 
            currentInput = self.modules[idModules - 1].output 
         end

         print('bp modules: ', idModules,self.modules[idModules], "currentGradInput\n", currentGradOutput:size(), "currentInput\n", currentInput:size())

         -- print(self.modules[idModules].modules)
         currentGradOutput = self:rethrowErrors(self.modules[idModules], idModules, 'backward', currentInput, currentGradOutput)
         assert(currentGradOutput:type() == currentInput:type())

         -- assert(parameters:type() == currentGradInput:type())
         -- if parameters:dim() ~= 0 then 
         --   self:rethrowErrors(self.modules[i], i, 'accGradParameters', currentInput, currentGradInput, 1)
         -- end
      end -- end backward for all modules

      self.gradInput = self.gradInput + currentGradOutput
      self.step = self.step + 1

      ---- [[ end backward ]] ----

   end -- ending idBufferStep -- 
   assert(self.step == self.bufferSize + 1, 'get step: '..self.step )
      assert(self.output:dim() == 5)
print('output in: ', self.output:size())
   return self.output, accErr, self.gradInput
end
--[[
function RecursiveSequential:backward(input, target)
   local accErr = 0
   local accGradInput = torch.Tensor():resizeAs(input):fill(0)
   self.gradInput = torch.Tensor():resizeAs(input):fill(0)
  
   -- start idBufferStep -- 
   for idBufferStep = 1, self.bufferSize do
        -- store output of currentBufferStep
      local currentOutput = self.output[idBufferStep]
      if idBufferStep == 1 then currentInput = input else currentInput = self.output[idBufferStep - 1]
      local currentInput = currentInput:float()

      -- get error and gradOutput
      local currentTarget = target[idBufferStep]:float()

      -- accumulating all the errors for all step:
      accErr = accErr + self.criterion:forward(currentOutput, currentTarget)

      local currentGradOutput = self.criterion:backward(currentOutput, currentTarget)

      -- get gradOutput and accGradParameters
      local currentGradInput = currentGradOutput
      for idModules = #self.modules,1, -1 do
         print('bp modules: ', i)
         -- print(self.modules[idModules])
         local parameters, gradParameters = self.modules[idModules]:getParameters()
         parameters = parameters:float()
         gradParameters = gradParameters:float()
         if idModules == 1 then currentInput = input else currentInput = self.modules[idModules - 1].output end
         -- print(self.modules[idModules].modules)
         currentGradInput = self:rethrowErrors(self.modules[idModules], i, 'backward', currentInput, currentGradInput)
         assert(currentGradInput:type() == currentInput:type())

         -- assert(parameters:type() == currentGradInput:type())
         if parameters:dim() ~= 0 then 
         --   self:rethrowErrors(self.modules[i], i, 'accGradParameters', currentInput, currentGradInput, 1)
         end
      end
      accGradInput = accGradInput + currentGradInput


      self.step = self.step + 1
   end
    -- end idBufferStep --

   self.gradInput = accGradInput

   return self.gradInput, accErr
 end
]]--
function RecursiveSequential:start()
   self.step = 1
   self.gradInput = torch.Tensor()
   self.gradOutput = torch.Tensor()
   self.output = torch.Tensor()
end


function RecursiveSequential:forward(input, outputSeqLength)
   -- WARNING! only use when no backward needed
   assert(torch.isTensor(inputBatch))
   assert(input:dim() == 4, 'fail input:dim() == 4')
   assert(target:dim() == 5, 'fail: target:dim() == 5')
   assert(target:size(1) == self.bufferSize, 'target dimension(1)'..target:size(1)..' mismatch bufferSize'..self.bufferSize)
   assert(target:size(2) == input:size(1)) -- batchSize
   -- assert(target:size(3) == input:size(2)) -- input/outputsize
   assert(target:size(4) == input:size(3)) -- height
   assert(target:size(5) == input:size(4)) -- width
   self:getDimension(input, target:size(3))
   local accErr = 0

   for t = 1, self.bufferSize do
      local currentOutput
      if t == 1 then
         currentOutput = input
      else 
         currentOutput = self.output[t - 1]
      end

      assert(currentOutput, 'lack currentOutput')
      for i = 1, #self.modules do
         currentOutput = self.rethowErrors(self.modules[i], i, 'updateOutput', currentOutput)
      end
      -- store output
      self.output[i] = currentOutput
   end
end

function RecursiveSequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.RecursiveSequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end




