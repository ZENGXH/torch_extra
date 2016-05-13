--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kernelSizeIn  - convolutional filter size to convolve input
  kernelSizeMem  - convolutional filter size to convolve cell; usually kernelSizeMem > kernelSizeIn  
--]]


--[[
  carefully handle the size of different variable:
  
  input: [seqLength x batchSize, inputSize, height, width]
  output: [seqLength x batchSize, outputSize, height, width]
  self.cells = [seqLength x batchSize, outputSize, height, width]

  updateOutput:
    1. reshape input and output:
    unpackBufferInput to be [seqLength, batchSize, inputSize, height, width]
    unpackBufferOutput to be [seqLength, batchSize, outputSize, height, width]
    self:unpackBuffer self.cells to be 
    2. passthough LSTM instance by instance:
      for idSeq = 1, seqLength do
        self.module: 
          -- local stepInput = input[idSeq], size: [batchSize, inputSize, height, width]
          -- local stepOutput: [batchSize, outputSize, height, width]
          if idSeq ~= 1 then
            {self.output[idSeq] , self.cells[idSeq]} = self.module:forward({input[idSeq], self.cells[idSeq - 1], self.output[idSeq - 1]})
          else
            {self.output[idSeq] , self.cells[idSeq]} = self.module:forward({input[idSeq], self.initCell, self.initOutput})
          end
      end

    3. reshapeback input and output:
    packBufferInput and output to be:
        input: [seqLength x batchSize, inputSize, height, width]
        self.output: [seqLength x batchSize, outputSize, height, width]
        self.cells
  end

  gradOutput: [seqLength x batchSize, outputSize, height, width]
  input: [seqLength x batchSize, inputSize, height, width]
  gradInput: [seqLength x batchSize, inputSize, height, width]

  < backward >
    1. reshape input and output:
    unpackBufferInput to be [seqLength, batchSize, inputSize, height, width]
    unpackBufferOutput to be [seqLength, batchSize, outputSize, height, width]
    unpackBufferGradOutput to be [seqLength, batchSize, outputSize, height, width]
    unpackBufferGradInput to be [seqLength, batchSize, outputSize, height, width]
    unpackBufferCell

    2. passthough LSTM instance by instance:
      for idSeq = 1, seqLength do
        self.module: 
          local stepInput = input[idSeq], size: [batchSize, inputSize, height, width]
          local stepOutput: [batchSize, outputSize, height, width]
          {gradInput[idSeq], gradCell} = self.module:backward({input[idSeq], self.cells[idSeq - 1], self.output[idSeq - 1]}, {gradOutput[idSeq], gradCell})
      end

    3. reshapeback input and output:
      self:packBuffer:
        input: [seqLength x batchSize, inputSize, height, width]
        self.output: [seqLength x batchSize, outputSize, height, width]
        gardOutput
        self.gardInput
        self.cell
  end

--]]

--[[
   StepConvLSTM need to work with nn.Sequencer(nn.stepStepConvLSTM)
   it will only remember one cell state before, 

   # TODO: if need more previous cell, output it and save
   it will only remember one gradInput as Tensor type instead of table
   all outputs will be saved in nn.Sequencer

   the reason StepConvLSTM do not inheritance from AbstractRecurrent is that
   ConvLSTM cost outofmemory problem during training and we only have 5G memory
   it will be better if we have a LSTM keep less things in memory..

   the model building part are inspied from 
   https://github.com/jcjohnson/torch-rnn/blob/master/LSTM.lua

--]]
local _ = require 'moses'
require 'nn'
require 'dpnn'
require 'rnn'
require 'optim'
-- require 'extracunn'
-- torch.setdefaulttensortype('torch.FloatTensor')
local StepConvLSTM, parent = torch.class('nn.StepConvLSTM', 'nn.Container')

function StepConvLSTM:__init(inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width, defaultType)
    -- print(inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width, defaultType)
    assert(defaultType, 'defaultType required')
    assert(batchSize, 'input are required in batchMode')    -- to reduce the complexity
    assert(height, 'we are sad about this, but fast lstm need size of image ')
    assert(width, 'we are sad about this, but fast lstm need size of image ')
    assert(inputSize)
    assert(outputSize)
    assert(bufferStep)
    assert(kernelSizeIn)
    assert(kernelSizeMem)
    assert(stride)
    assert(height)
    assert(width)
    parent.__init(self)
    self.defaultType = defaultType
    self.inputSize = inputSize
    self.outputSize = outputSize
    self.bufferStep = bufferStep 
    -- the gates will store the #bufferStep data, also the maximum step for backward

    -- self.gates = torch.Tensor() -- will be [batch/
    self.kernelSizeIn = kernelSizeIn or 3
    self.kernelSizeMem = kernelSizeMem or 3
    self.padIn = torch.floor(self.kernelSizeIn/2)
    self.padMem = torch.floor(self.kernelSizeMem/2)

    self.stride = stride or 1
    self.batchSize = batchSize or 1 

    self.height = height
    self.width = width
    self.step = 1
    -- self.modules = {}
    self.module = self:buildModel()-- :float()

    self.initCell = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)
    self.initOutput = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)
    self.lastCell = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)
    self.lastOutput = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)
    self.lastGradInput = torch.Tensor(self.batchSize, self.inputSize, self.height, self.width):fill(0)

    self.gradPrevCell = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)

    self.output = torch.Tensor(self.bufferStep * self.batchSize, self.outputSize, self.height, self.width):fill(0)
    -- print('self output init: ', self.output:size())
    self.gradInput = torch.Tensor(self.bufferStep * self.batchSize, self.inputSize, self.height,self.width):fill(0)
    self.cells = torch.Tensor(self.bufferStep * self.batchSize, self.outputSize, self.height, self.width):fill(0)

    --print('configure StepConvLSTM:')
    --print('inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width')
    --print(self.inputSize, self.outputSize, self.bufferStep, self.kernelSizeIn, self.kernelSizeMem, self.stride, self.batchSize, self.height, self.width)

    self:forget()

end



-- put them in self for convinence of debugging



function StepConvLSTM:buildModel()
    i2g = nn.SpatialConvolution(self.inputSize, self.outputSize*4, 
                                  self.kernelSizeIn, self.kernelSizeIn, 
                                  self.stride, self.stride, 
                                  self.padIn, self.padIn)
    o2g = nn.SpatialConvolution(self.outputSize, self.outputSize*4, 
                                  self.kernelSizeMem, self.kernelSizeMem, 
                                  self.stride, self.stride, 
                                  self.padMem, self.padMem)
    w_init(i2g)
    w_init(o2g)
   if self.usenngraph then
      print('usting nngraph')
      return self:nngraphModel(i2g, o2g)
   end

      print('not using nngraph')
      local para = nn.ParallelTable():add(i2g):add(o2g)

      gates = nn.Sequential()
      gates:add(nn.NarrowTable(1,2))
      gates:add(para)
      gates:add(nn.CAddTable())

      gates:add(nn.Reshape(self.batchSize, 4, self.outputSize, self.height, self.width)) 
      gates:add(nn.SplitTable(2))
      transfer = nn.ParallelTable()
      transfer:add(nn.Sigmoid()):add(nn.Tanh()):add(nn.Sigmoid()):add(nn.Sigmoid())
      gates:add(transfer)

      local concat = nn.ConcatTable()
      concat:add(gates):add(nn.SelectTable(3))
      local seq = nn.Sequential()
      seq:add(concat)
      seq:add(nn.FlattenTable()) -- input, hidden, forget, output, cell
      
      -- input gate * hidden state
      local hidden = nn.Sequential()
      hidden:add(nn.NarrowTable(1,2))
      hidden:add(nn.CMulTable())
      
      -- forget gate * cell
      local cell = nn.Sequential()
      local concat = nn.ConcatTable()
      concat:add(nn.SelectTable(3)):add(nn.SelectTable(5))
      cell:add(concat)
      cell:add(nn.CMulTable())
      
      local nextCell = nn.Sequential()
      local concat = nn.ConcatTable()
      concat:add(hidden):add(cell)
      nextCell:add(concat)
      nextCell:add(nn.CAddTable())
      
      local concat = nn.ConcatTable()
      concat:add(nextCell):add(nn.SelectTable(4))
      seq:add(concat)
      seq:add(nn.FlattenTable()) -- nextCell, outputGate

      local cellAct = nn.Sequential()
      cellAct:add(nn.NarrowTable(1,2))
      local concat_output = nn.ConcatTable()
      concat_output:add(nn.Sequential():add(nn.SelectTable(1)):add(nn.Tanh()))
                    :add(nn.SelectTable(2))
      -- concat_output:add(nn.Tanh())
      cellAct:add(concat_output):add(nn.CMulTable())
      local concat = nn.ConcatTable()
      concat:add(cellAct):add(nn.SelectTable(1))

      seq:add(concat) -- :add()      

      return seq
end

function StepConvLSTM:nngraphModel(i2g, o2g)
  require 'nngraph'
  assert(nngraph, "Missing nngraph packBufferage")

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- prev_h[L]
  table.insert(inputs, nn.Identity()()) -- prev_c[L]

  -- local x, prev_h, prev_c = self:unpackBuffer(inputs)
  local x, prev_h, prev_c = inputs[1], inputs[2], inputs[3]
  -- evaluate the input sums at once for efficiency

  local i2h = i2g(x):annotate{name='i2h'}
  local h2h = o2g(prev_h):annotate{name='h2h'}

  local all_input_sums = nn.CAddTable()({i2h, h2h})
  local reshaped = nn.Reshape(4, self.batchSize, self.outputSize, self.height, self.width)(all_input_sums)

  -- local reshaped = nn.Reshape(4 * self.inputSize, self.outputSize, self.height, self.width)(all_input_sums)
  -- input, hidden, forget, output
  local n1, n2, n3, n4 = nn.SplitTable(1)(reshaped):split(4)
  local in_gate = nn.Sigmoid()(n1)
  local in_transform = nn.Tanh()(n2)
  local forget_gate = nn.Sigmoid()(n3)
  local out_gate = nn.Sigmoid()(n4)

  -- perform the LSTM update
  local next_c           = nn.CAddTable()({
   nn.CMulTable()({forget_gate, prev_c}),
   nn.CMulTable()({in_gate,     in_transform})
  })
  -- gated cells form the output
  local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  local outputs = {next_h, next_c}
  mlp = nn.gModule(inputs, outputs)

  return mlp
end


--[[

`forward`: updateOutput, take input, check if there is userPrevOutput, ,

'backward':
pass the parameters to previous layer: 

`updateGradInput`: 
  for LSTM 1, input is: tensor x,
  for self.module, input is {x, prevOutput, prevCell} 

--]]
function StepConvLSTM:typeChecking(para, msg)
  if torch.type(para) == 'table' then
    self.typeChecking(para[1], msg)
  elseif(torch.isTensor(para)) then
    msg = 'fail TypeChecking on'..msg
    assert(para:type() == self.defaulType, msg)
  elseif(torch.isTypeOf(para, 'nn.Container')) or para.modules then
    for i = 1, #para.modules do
      self.typeChecking(para.modules[i])
    end
  elseif(torch.isTypeOf(para, 'nn.Module')) then
    assert(para._type == defaulType)
  end
end

function StepConvLSTM:unpackBuffer(x, bufferStepDim)
  assert(x:dim() == 4, 'size: '..x:size(1)..' '..x:size(2)..' '..x:size(3)..' '..x:size(4))
  local size = x:size(2)
  --print(bufferStepDim)  
  assert(size == self.inputSize or size == self.outputSize, 'dimention2 of variable not input/output size, get size:'..x:size(1)..', '..x:size(2)..', '..x:size(3)..', '..x:size(4))
  local bufferStepDim = bufferStepDim or self.bufferStep

  x = x:view(bufferStepDim, self.batchSize, size, self.height, self.width)
  return x
end

function StepConvLSTM:packBuffer(x, bufferStepDim)
  assert(x:dim() == 5)
  local size = x:size(3)
  local bufferStepDim = bufferStepDim or self.bufferStep
  x = x:view(bufferStepDim * self.batchSize, size, self.height, self.width)
  return x
end

function StepConvLSTM:updateOutput(input)
  -- local p, g = self.module:getParameters()
  -- assert(g:mean() == 0, 'gradient not zero at the begin'..g:mean())
  assert(torch.isTensor(input))
  input = self:unpackBuffer(input)
  self.output = self:unpackBuffer(self.output)
  self.cells = self:unpackBuffer(self.cells)

  if(self.step > self.bufferStep) then
    self.step = 1
    print('ConvLSTM reset step to 1')
  end

  assert(input:dim() == 5, 'input dimension, 5 required get'..input:dim())
  assert(input:size(1) == self.bufferStep)
  assert(input:size(2) == self.batchSize, 'inputBatch dim2 = batchSize require')
  assert(input:size(3) == self.inputSize)
  assert(input:size(4) == self.height)
  assert(input:size(5) == self.width) 

  local outputTable = {}

  for idStep = 1, self.bufferStep do
    -- print('StepConvLSTM sequencing step: ', idStep)
    assert(self.output:type() == input:type(), 'fail self.output:type() == input:type()')
    assert(input:type() == self.module._type, 'input:type() != self.module._type 1:'..input:type()..' and 2: '..self.module._type)
    --assert(self.prevOutput:size(1) == self.batchSize,'fail self.prevOutput:size(1) == self.batchSize' )
    --assert(self.prevOutput:size(2) == self.outputSize)
    --assert(self.prevOutput:size(3) == self.height)
    --assert(self.prevOutput:size(4) == self.width)
    --assert(self.module._type == self.zeroTensor:type(), 'fail: self.module._type == self.zeroTensor:type()')
    --assert(self.prevOutput:type() == input:type(), self.prevOutput:type())
    --assert(self.prevCell:type() == input:type(), self.prevCell:type())
    --assert(self.module._type == input:type(), 'self.module._type != input:type()')
    --assert(torch.isTypeOf(self.module, 'nn.Module'))

    -- self.recursiveTypeChecking(self.module, 'torch.FloatTensor')
    -- print({input[idStep], self.initCell, self.initOutput})
    -- print(self.module)
    if idStep ~= 1 then
      outputTable = self.module:updateOutput({input[idStep], self.cells[idStep - 1], self.output[idStep - 1]})
    else
      outputTable = self.module:updateOutput({input[idStep], self.initCell, self.initOutput})
    end
    --print('outputTable', outputTable)
    --print({self.output, self.cells})
    self.output[idStep] = outputTable[1]
    self.cells[idStep] = outputTable[2]
    self.step = self.step + 1
  end
  -- print('end StepConvLSTM')
  self.lastCell = self.cells[self.bufferStep]
  self.lastOutput = self.output[self.bufferStep]

  input = self:packBuffer(input)
  self.output = self:packBuffer(self.output)
  self.cells = self:packBuffer(self.cells)

  -- note that we don't return the cell, just the output
  return self.output
end

-- forget from empty state
function StepConvLSTM:forget()
  -- print('calling forget')
  self.step = 1

  self.initCell = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)
  self.initOutput = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):fill(0)
  self.output = torch.Tensor(self.bufferStep * self.batchSize, self.outputSize, self.height, self.width):fill(0)
  self.gradInput = torch.Tensor(self.bufferStep * self.batchSize, self.inputSize, self.height,self.width):fill(0)
  self.cells = torch.Tensor(self.bufferStep * self.batchSize, self.outputSize, self.height, self.width):fill(0)

end

function StepConvLSTM:getDimensionForGradInput(input)
-- if input dimension = 4, gradInput has the same size
-- if input diemnsion = 5, gradInput size is the last 4 dim
  local batchSize
  local inputSize
  local height
  local width
  assert(input:dim() == 5 or input:dim() == 4, 'input dimension need to be 4 or 5, get '..input:dim())
  if(input:dim() == 5) then
    batchSize = input:size(2)
    inputSize = input:size(3)
    height = input:size(4)
    width = input:size(5)
  elseif(input:dim() == 4) then
    batchSize = input:size(1)
    inputSize = input:size(2)
    height = input:size(3)
    width = input:size(4)
  end

  -- self.gradInput = torch.Tensor(batchSize, inputSize, height, width):fill(0)

end

function StepConvLSTM:maxBackWard(input, gradOutput, scale)
  --[[maxBackWard can calculate the step for bp automatically,
      for normal bp: gradOutput: (self.bufferStep*self.batchSize, self.outputSize, self.height, self.width)
      for maxBackward(maxiBpStep*self.batchSize, self.outputSize, self.height, self.width)
  ]]--
  assert(self.bufferStep~=1, 'maxiBp only apply for bufferStep ~= 1 StepConvLSTM')

  local maxiBpStep = gradOutput:size(1)/self.batchSize
  --print('bufferStepDim: ', maxiBpStep)

  assert(self.bufferStep - maxiBpStep > 1, 'maxiBp should at least 2 step smaller than the bufferStep')
  -- print('maxi bp step is', maxiBpStep)
  local scale = scale or 1
  gradOutput = self:unpackBuffer(gradOutput, maxiBpStep)
  input = self:unpackBuffer(input)
  self.gradInput = self:unpackBuffer(self.gradInput)
  self.cells = self:unpackBuffer(self.cells)
  self.output = self:unpackBuffer(self.output)

  assert(gradOutput:dim() == 5, 'gradOutput dimension = 5 required')
  assert(input:dim() == 5, 'input dimension = 5 required')
  local gradTable = {}
  -- ************ only apply for bufferStep ~= 1 StepConvLSTM ************
  
  for idStep = self.bufferStep, self.bufferStep - maxiBpStep + 1, -1 do
    -- print('idStep:', idStep)
    -- print(idStep - (self.bufferStep - maxiBpStep) + 1)
    -- print('size',gradOutput:size())
    gradTable = self.module:backward({input[idStep], self.cells[idStep - 1], self.output[idStep - 1]}, {gradOutput[idStep - (self.bufferStep - maxiBpStep)], self.gradPrevCell}, scale)
    self.gradInput[idStep] = gradTable[1]
    self.gradPrevCell = gradTable[2]
    gradPrevOutput = gradTable[3]
    self.step  = self.step - 1
  end
  
  -- ************
  -- self.lastGradInput = self.gradInput[1]
  -- self.lastGradPrevOutput = gradPrevOutput

  input = self:packBuffer(input)
  gradOutput = self:packBuffer(gradOutput, maxiBpStep)
  self.gradInput = self:packBuffer(self.gradInput)
  self.cells = self:packBuffer(self.cells)
  self.output = self:packBuffer(self.output)
  -- print('maxiBp done')
  return self.gradInput  -- 5d
end

function StepConvLSTM:backward(input, gradOutput, scale)
  local scale = scale or 1
  gradOutput = self:unpackBuffer(gradOutput)
  input = self:unpackBuffer(input)
  self.gradInput = self:unpackBuffer(self.gradInput)
  self.cells = self:unpackBuffer(self.cells)
  self.output = self:unpackBuffer(self.output)

  assert(gradOutput:dim() == 5, 'gradOutput dimension = 5 required')
  assert(input:dim() == 5, 'input dimension = 5 required')
  local gradTable = {}
  local gradPrevOutput
  -- ****************************
  if self.bufferStep ~= 1 then -- move 'if' out of for loop
    for idStep = self.bufferStep, 2, -1 do
      gradTable = self.module:backward({input[idStep], self.cells[idStep - 1], self.output[idStep - 1]}, {gradOutput[idStep], self.gradPrevCell}, scale)
      self.gradInput[idStep] = gradTable[1]
      self.gradPrevCell = gradTable[2]
      gradPrevOutput = gradTable[3]
      self.step  = self.step - 1
    end
  end
  --print('input size', {input[1], self.initCell, self.initOutput})
  --print('grad size', {gradOutput[1], self.gradPrevCell})
  --print('module output', self.module.output)
  gradTable = self.module:backward({input[1], self.initCell, self.initOutput}, {gradOutput[1], self.gradPrevCell}, scale)
  self.gradInput[1] = gradTable[1]
  self.gradPrevCell = gradTable[2]
  gradPrevOutput = gradTable[3]
  self.step  = self.step - 1
  -- ****************************
  self.lastGradInput = self.gradInput[1]
  self.lastGradPrevOutput = gradPrevOutput

  input = self:packBuffer(input)
  gradOutput = self:packBuffer(gradOutput)
  self.gradInput = self:packBuffer(self.gradInput)
  self.cells = self:packBuffer(self.cells)
  self.output = self:packBuffer(self.output)
  return self.gradInput  -- 5d
end


function StepConvLSTM:updateGradInput(input, gradOutput, scale)
  local scale = scale or 1
  self:unpackBuffer(input)
  self:unpackBuffer(gradOutput)
  self:unpackBuffer(self.gradInput)
  self:unpackBuffer(self.cells)
  self:unpackBuffer(self.output)

  assert(gradOutput:dim() == 5, 'gradOutput dimension = 5 required')
  assert(input:dim() == 5, 'input dimension = 5 required')
  local gradTable = {}
  local gradPrevOutput
  -- ************************
  if self.bufferStep ~= 1 then -- move 'if' out of for loop
    for idStep = self.bufferStep, 2, -1 do
      gradTable = self.module:updateGradInput({input[idStep], self.cells[idStep - 1], self.output[idStep - 1]}, {gradOutput[idStep], self.gradPrevCell}, scale)
      self.gradInput[idStep] = gradTable[1]
      self.gradPrevCell = gradTable[2]
      gradPrevOutput = gradTable[3]
      self.step  = self.step - 1
    end
  end
  gradTable = self.module:updateGradInput({input[1], self.initCell, self.initOutput}, {gradOutput[1], self.gradPrevCell}, scale)
  self.gradInput[idStep] = gradTable[1]
  self.gradPrevCell = gradTable[2]
  gradPrevOutput = gradTable[3]
  self.step  = self.step - 1
  -- ************************

  self:packBuffer(input)
  self:packBuffer(gradOutput)
  self:packBuffer(self.gradInput)
  self:packBuffer(self.cells)
  self:packBuffer(self.output)
  return self.gradInput  -- 5d
end

function StepConvLSTM:accGradParameters(input, gradOutput, lr)
  local lr = lr or 1
  self:unpackBuffer(input)
  self:unpackBuffer(gradOutput)
  self:unpackBuffer(self.gradInput)
  self:unpackBuffer(self.cells)
  self:unpackBuffer(self.output)

  assert(gradOutput:dim() == 5, 'gradOutput dimension = 5 required')
  assert(input:dim() == 5, 'input dimension = 5 required')
  local gradTable = {}
  local gradPrevOutput
  
  -- ************************
  if self.bufferStep ~= 1 then -- move 'if' out of for loop
    for idStep = self.bufferStep, 2, -1 do
      self.module:accGradParameters({input[1], self.cells[idStep - 1], self.output[idStep - 1]}, {gradOutput[idStep], self.gradPrevCell}, scale)
      self.gradPrevCell = self.module.gradInput[2]
    end
  end
  self.module:accGradParameters({input[1], self.initCell, self.initOutput}, {gradOutput[1], self.gradPrevCell}, scale)
  -- *************************

  self:packBuffer(input)
  self:packBuffer(gradOutput)
  self:packBuffer(self.gradInput)
  self:packBuffer(self.cells)
  self:packBuffer(self.output)
  return 
end

-- ===============================================================
function StepConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
end


function StepConvLSTM:zeroGradParameters()
  self.module:zeroGradParameters()
end
--[[
function StepConvLSTM:__tostring__()
  print('configure StepConvLSTM:')
  print('inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width')
  print(self.inputSize, self.outputSize, self.bufferStep, self.kernelSizeIn, self.kernelSizeMem, self.stride, self.batchSize, self.height, self.width)
  print("print modules[1] of the StepConvLSTM:")
  print(self.module)
end]]--
