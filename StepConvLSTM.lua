--[[
  Convolutional LSTM for short term visual cell
  inputSize - number of input feature planes
  outputSize - number of output feature planes
  rho - recurrent sequence length
  kernelSizeIn  - convolutional filter size to convolve input
  kernelSizeMem  - convolutional filter size to convolve cell; usually kernelSizeMem > kernelSizeIn  
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

    self.gates = torch.Tensor() -- will be [batch/
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
    -- self.module = self.module
    -- local D, H, height, width = self.inputSize, self.outputSize, self.height, self.width
    -- pre-allocate all the parameters
    -- self.weight = torch.Tensor(D + H, 4 * H) 
    -- self.gradWeight = torch.Tensor(D + H, 4 * H):zero()

    -- self.bias = torch.Tensor(4 * H)
    -- self.gradBias = torch.Tensor(4 * H):zero()
    -- self:weightInit(method)
    self.userPrevCell = torch.Tensor(batchSize, outputSize, height, width)
    self.userPrevOutput = torch.Tensor(batchSize, outputSize, height, width)
    self.prevOutput = torch.Tensor(batchSize, outputSize, height, width)
    self.prevCell = torch.Tensor(batchSize, outputSize, height, width)    -- This will be (N, T, H)
    self.gates = torch.Tensor()   -- This will be (N, T, 4H)
    -- self.buffer1 = torch.Tensor() -- This will be (N, H)
    -- self.buffer2 = torch.Tensor() -- This will be (N, H)
    -- self.buffer3 = torch.Tensor() -- This will be (1, 4H)
    -- self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)

    -- self.h0 = torch.Tensor()
    -- self.c0 = torch.Tensor()
    -- self.remember_states = false

    -- self.grad_c0 = torch.Tensor()
    -- self.grad_h0 = torch.Tensor()
    -- self.grad_x = torch.Tensor()
    -- self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
    self.output = torch.Tensor(outputSize, batchSize, height, width)
    self.zeroTensor = torch.Tensor(self.batchSize, self.outputSize, self.height, self.width):zero()
    self.outputs = torch.Tensor(bufferStep, batchSize, outputSize, height, width):fill(0)
    self.cells = torch.Tensor(bufferStep, batchSize, outputSize, height, width):fill(0)
    print('configure StepConvLSTM:')
    print('inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width')
    print(self.inputSize, self.outputSize, self.bufferStep, self.kernelSizeIn, self.kernelSizeMem, self.stride, self.batchSize, self.height, self.width)

    self:forget()
     ---
end

function StepConvLSTM:clearState()
  print('clear state of LSTM:')
  -- normal clear:
  self.module:clearState()
  assert(self.module.output ~= nil)
  assert(self.module.gradInput ~= nil)

  -- forget previous
  self.prevOutput = self.prevOutput:new()
  self.prevCell = self.prevCell:new()

  -- TODO: COLLECT GABAGE?
  -- TODO: ANYTHING else?
end

function StepConvLSTM:setFloat()
  self.module = self.module:float()
  print(self.module)
  assert(self.module._type == 'torch.FloatTensor')
end
function StepConvLSTM:weightInit(method)
  local method = method or nil
  if not method then
    method = 'xiva...?'
  else
    method = 'gaussian'
  end
  -- TODO:...
  print('not implement: weightInit')
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
  assert(nngraph, "Missing nngraph package")

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- prev_h[L]
  table.insert(inputs, nn.Identity()()) -- prev_c[L]

  local x, prev_h, prev_c = unpack(inputs)

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




function StepConvLSTM:updateOutput(inputBatch)

  if(self.step > self.bufferStep) then
    self.step = 1
    print('ConvLSTM reset step to 1')
  end
  print('inputBatch size: ' , inputBatch:size())
  assert(inputBatch:dim() == 5, 'input dimension, 5 required get'..inputBatch:dim())
  print("size is ",inputBatch:size(1), self.bufferStep)
  assert(inputBatch:size(1) == self.bufferStep)
  assert(inputBatch:size(2) == self.batchSize)
  assert(inputBatch:size(3) == self.inputSize)
  assert(inputBatch:size(4) == self.height)
  assert(inputBatch:size(5) == self.width) 

  for t = 1, self.bufferStep do
    print('StepConvLSTM sequencing step: '..t)
    input = inputBatch[t]
    input = input:type(self.defaultType)
    assert(self.outputs:type() == input:type(), 'fail self.outputs:type() == input:type()')
    -- print(self.module)
    assert(input:type() == self.module._type, 'input:type() != self.module._type 1:'..input:type()..' and 2: '..self.module._type)

    -- self.userPrevCell = self.userPrevCell or self.zeroTensor
    -- self.userPrevOutput = self.userPrevOutput or self.zeroTensor

    if(not torch.isTensor(input)) then
      print('StepConvLSTM input ', input)
    end
     
    if self.step == 1 then
      assert(self.zeroTensor:size(1) == self.batchSize)
      self.prevCell = self.zeroTensor
      self.prevOutput = self.zeroTensor 
    else
      assert(self.outputs:type() == input:type(), 'fail, self.outputs:type() == input:type()')
      self.prevCell = self.cells[self.step - 1] -- self.cells[{{self.step - 1}, {},{},{},{}}] -- T, B, H, h, w -> B, H, h, w
      assert(self.prevCell:dim() == 4)
      self.prevOutput = self.outputs[self.step - 1] -- self.outputs[{{self.step - 1}, {},{},{},{}}]
        -- local cur_gates = self.pregate:forward({input, prevOutput})-- self.gates[{{self.step - 1},{},{},{},{}] -- gates is T,B,4*H,h,w
        -- i = cur_gates[{{self.step}, }]
        -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
      assert(self.outputs:type() == input:type(), 'fail after, self.outputs:type() == input:type()')
    end
      

      -- print('p2')
      --[[
      assert(input:dim() == 4,'input dimenstion should be 4')
      -- print("size is ",input:size(1), self.bufferStep)
      assert(input:size(1) == self.batchSize)
      assert(input:size(2) == self.inputSize)
      assert(input:size(3) == self.height)
      assert(input:size(4) == self.width)

      ]]--
      -- assert(self.prevOutput)
      -- print('p3')
      -- print('self.prevOutput is', self.prevOutput)
      -- print(self.prevOutput:size())
      assert(self.prevOutput:size(1) == self.batchSize,'fail self.prevOutput:size(1) == self.batchSize' )
      assert(self.prevOutput:size(2) == self.outputSize)
      assert(self.prevOutput:size(3) == self.height)
      assert(self.prevOutput:size(4) == self.width)
      assert(self.module._type == self.zeroTensor:type(), 'fail: self.module._type == self.zeroTensor:type()')
      assert(self.prevOutput:type() == input:type(), self.prevOutput:type())
      assert(self.prevCell:type() == input:type(), self.prevCell:type())
      assert(self.module._type == input:type(), 'self.module._type != input:type()')
      -- print('p4.1', self.module.modules[1].output)
      -- assert(defaulType, 'need to specify defaulType globally')
      self.module:type(self.defaultType)

      -- print('p4', self.module.modules[2]._type)
      assert(torch.isTypeOf(self.module, 'nn.Module'))
      assert(torch.isTypeOf(self.module, 'nn.Container'))

      -- self.recursiveTypeChecking(self.module, 'torch.FloatTensor')
      local outputAndCellTable = self.module:updateOutput({input, self.prevOutput, self.prevCell})
      local output = outputAndCellTable[1]
      local cell = outputAndCellTable[2]

      assert(self.outputs:type() == input:type(), 'fail self.outputs:type() == input:type()')
      assert(output:type() == input:type(), 'fail output:type() == input:type()')
      assert(cell:type() == input:type(), 'fail output:type() == input:type()')
      assert(cell)
      self.cells[self.step] = cell
      self.outputs[self.step] = output

      assert(self.outputs:type() == input:type(), self.outputs:type())
      self.output = output
      self.step = self.step + 1
  end

  -- note that we don't return the cell, just the output
  return self.outputs
end

-- forget from empty state
function StepConvLSTM:forget()
  print('calling forget')
  -- in case they are not assighed
  -- defaulType = self.zeroTensor:float() -- :type(defaulType)
  -- print(defaulType)
  assert(torch.isTensor(self.zeroTensor), 'self.zeroTensor is not a tensor')
  -- print(self.defaultType)
  self.prevCell = self.zeroTensor:type(self.defaultType)
  self.prevOutput = self.zeroTensor:type(self.defaultType)
  self.gradPrevCell = self.zeroTensor:type(self.defaultType)
  self.gradPrevOutput = self.zeroTensor:type(self.defaultType)
  -- print(self.prevCell)
  -- print(self.zeroTensor)
  assert(torch.isTensor(self.prevCell), 'self.prevCell is not a tensor: ')
  -- self.cells = torch.Tensor(self.bufferStep, self.batchSize, self.outputSize, self.height, self.width)
  -- self.outputs = torch.Tensor(self.bufferStep, self.batchSize,self.outputSize, self.height, self.width)
  -- self.prevOutput = self.zeroTensor
  -- self.prevCell = self.zeroTensor
  -- self.output = torch.Tensor(self.outputSize, self.batchSize,self.height, self.width)
  -- self.module:zeroGradParameters()
  -- self.module:forget()
  self.step = 1
end

--TODO: forget from previous state -> use for online version


function StepConvLSTM:backward(input, gradOutput, scale)
  -- input
  if self.step == 1 then
    self.prevCell = self.zeroTensor
    self.prevOutput = self.zeroTensor
    self.zeroTensor:resize(self.batchSize, self.outputSize, input:size(3), input:size(4)):zero()
  else
    self.prevCell = self.cells[self.step - 1]
    self.prevOutput = self.outputs[self.step - 1]
  end

  -- gradOutput of modules

  if self.step ~= self.bufferStep then
    self.gradPrevCell = self.gradPrevCell or self.zeroTensor
  end

  self.gradPrevCell = self.gradPrevCell or gradOutput:new():fill(0)
  scale = scale or 1

  local x = input
  -- print(self.gradPrevCell)
  -- assert(self.gradPrevCell)


  local gradTable = self.module:backward({input, self.prevOutput, self.prevCell}, {gradOutput, self.gradPrevCell})
  
  self.gradInput = gradTable[1]
  self.gradPrevOutput = gradTable[1]
  self.gradPrevCell = gradTable[2]

  --  print(gradInput_x)
  -- gradInput_x, gradInput_cell, gradInput_outputs = , 

  self.module:accGradParameters({input, self.prevOutput, self.prevCell}, {gradOutput, self.gradPrevCell})
  -- self.module:updateParameters()
  
  self.step  = self.step - 1
  return self.gradInput 
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

--[[
function StepConvLSTM:__tostring__()
  print('configure StepConvLSTM:')
  print('inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width')
  print(self.inputSize, self.outputSize, self.bufferStep, self.kernelSizeIn, self.kernelSizeMem, self.stride, self.batchSize, self.height, self.width)
  print("print modules[1] of the StepConvLSTM:")
  print(self.module)
end]]--
