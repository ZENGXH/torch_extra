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
-- require 'extracunn'

local StepConvLSTM, parent = torch.class('nn.StepConvLSTM', 'nn.Container')

function StepConvLSTM:__init(inputSize, outputSize, rho, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width)
   parent.__init(self)
   self.inputSize = inputSize
   self.outputSize = outputSize

   self.kernelSizeIn = kernelSizeIn or 3
   self.kernelSizeMem = kernelSizeMem or 3
   self.padIn = torch.floor(self.kernelSizeIn/2)
   self.padMem = torch.floor(self.kernelSizeMem/2)

   self.stride = stride or 1
   assert(batchSize, 'input are required in batchMode')    -- to reduce the complexity
   self.batchSize = batchSize or 1 
   assert(height, 'we are sad about this, but fast lstm need size of image ')
   assert(width, 'we are sad about this, but fast lstm need size of image ')
   self.H = height
   self.W = width
end

function StepConvLSTM:prepareGate()
  self.i2g = nn.SpatialConvolution(self.inputSize, self.outputSize*4, 
                                  self.kernelSizeIn, self.kernelSizeIn, 
                                  self.stride, self.stride, 
                                  self.padIn, self.padIn)
  self.o2g = nn.SpatialConvolution(self.outputSize, self.outputSize*4, 
                                  self.kernelSizeMem, self.kernelSizeMem, 
                                  self.stride, self.stride, 
                                  self.padMem, self.padMem)  
end
-- put them in self for convinence of debugging



function StepConvLSTM:buildModel()
   prepareGate()
   if self.usenngraph then
      print('usting nngraph')
      return self:nngraphModel()
   else
      print('not using nngraph')
      local para = nn.ParallelTable():add(self.i2g):add(self.o2g)

      gates = nn.Sequential()
      gates:add(nn.NarrowTable(1,2))
      gates:add(para)
      gates:add(nn.CAddTable())

      -- Reshape to (batch_size, n_gates, hid_size)
      -- Then slize the n_gates dimension, i.e dimension 2
      --print('reshape:: ')
      --print(self.batchSize, 4, self.outputSize, self.H, self.W)
      -- print(self.batchSize, 4, self.outputSize, self.H, self.W)
      gates:add(nn.Reshape(self.batchSize, 4, self.outputSize, self.H, self.W)) 
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
      cellAct:add(nn.SelectTable(1))
      cellAct:add(nn.Tanh())
      local concat = nn.ConcatTable()
      concat:add(cellAct):add(nn.SelectTable(2))
      local output = nn.Sequential()
      output:add(concat)
      output:add(nn.CMulTable())
      
      local concat = nn.ConcatTable()
      concat:add(output):add(nn.SelectTable(1))
      seq:add(concat)

return seq



function FastConvLSTM:nngraphModel()
  assert(nngraph, "Missing nngraph package")

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- prev_h[L]
  table.insert(inputs, nn.Identity()()) -- prev_c[L]

  local x, prev_h, prev_c = unpack(inputs)

  -- evaluate the input sums at once for efficiency

  local i2h = self.i2g(x):annotate{name='i2h'}
  local h2h = self.o2g(prev_h):annotate{name='h2h'}

  local all_input_sums = nn.CAddTable()({i2h, h2h})

  local reshaped = nn.Reshape(4, self.outputSize, self.H, self.W)(all_input_sums)
  -- input, hidden, forget, output
  local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
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


function StepConvLSTM:updateOutput(input)
   local prevOutput, prevCell

   if(not torch.isTensor(input)) then
      print('StepConvLSTM input ', input)
   end

   if self.step == 1 then
      self.prevOutput = self.userPrevOutput or self.zeroTensor
      self.prevCell = self.userPrevCell or self.zeroTensor
      self.zeroTensor:resize(self.batchSize, self.outputSize, input:size(3), input:size(4)):zero()
      -- to be deleted, to reduce complexity all in batch mode
   else
      -- previous output and memory of this module
      -- prevOutput = self.output
      -- prevCell   = self.cell
   end
      
   -- output(t), cell(t) = lstm{input(t), output(t-1), cell(t-1)}
   local output, cell

   if self.train ~= false then
      -- self:recycle()
      -- local recurrentModule = self:getStepModule(self.step)
      if verbose then
         print('self.step of StepConvLSTM is ',self.step,recurrentModule)
      end
      -- the actual forward propagation
      output, cell = unpack(recurrentModule:updateOutput{input, prevOutput, prevCell})
   else
      output, cell = unpack(self.recurrentModule:updateOutput{input, prevOutput, prevCell})
   end
   
   self.outputs[self.step] = output
   self.cells[self.step] = cell
   
   self.output = output
   self.cell = cell
   
   self.step = self.step + 1
   self.gradPrevOutput = nil
   self.updateGradInputStep = nil
   self.accGradParametersStep = nil
   self.gradParametersAccumulated = false
   -- note that we don't return the cell, just the output
   return self.output
end

function StepConvLSTM:initBias(forgetBias, otherBias)
  local fBias = forgetBias or 1
  local oBias = otherBias or 0
  self.inputGate.modules[2].modules[1].bias:fill(oBias)
  self.outputGate.modules[2].modules[1].bias:fill(oBias)
  self.cellGate.modules[2].modules[1].bias:fill(oBias)
  self.forgetGate.modules[2].modules[1].bias:fill(fBias)
end



--- KEEP FOR DEBUGGING ONLY ---

function StepConvLSTM:buildGate()
   -- Note : Input is : {input(t), output(t-1), cell(t-1)}
   local gate = nn.Sequential()
   gate:add(nn.NarrowTable(1,2)) -- we don't need cell here
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kernelSizeIn, self.kernelSizeIn, self.stride, self.stride, self.padIn, self.padIn)
   local output2gate = nn.SpatialConvolution(self.outputSize, self.outputSize, self.kernelSizeMem, self.kernelSizeMem, self.stride, self.stride, self.padMem, self.padMem)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate) 
   gate:add(para)
   gate:add(nn.CAddTable())
   gate:add(nn.Sigmoid())
   return gate
end

function StepConvLSTM:buildInputGate()
   self.inputGate = self:buildGate()
   return self.inputGate
end

function StepConvLSTM:buildForgetGate()
   self.forgetGate = self:buildGate()
   return self.forgetGate
end

function StepConvLSTM:buildcellGate()
   -- Input is : {input(t), output(t-1), cell(t-1)}, but we only need {input(t), output(t-1)}
   local hidden = nn.Sequential()
   hidden:add(nn.NarrowTable(1,2))
   local input2gate = nn.SpatialConvolution(self.inputSize, self.outputSize, self.kernelSizeIn, self.kernelSizeIn, self.stride, self.stride, self.padIn, self.padIn)
   local output2gate = nn.SpatialConvolution(self.outputSize, self.outputSize, self.kernelSizeMem, self.kernelSizeMem, self.stride, self.stride, self.padMem, self.padMem)
   local para = nn.ParallelTable()
   para:add(input2gate):add(output2gate)
   hidden:add(para)
   hidden:add(nn.CAddTable())
   hidden:add(nn.Tanh())
   self.cellGate = hidden
   return hidden
end

function StepConvLSTM:buildcell()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.inputGate = self:buildInputGate() 
   self.forgetGate = self:buildForgetGate()
   self.cellGate = self:buildcellGate()
   -- forget = forgetGate{input, output(t-1), cell(t-1)} * cell(t-1)
   local forget = nn.Sequential()
   local concat = nn.ConcatTable()
   concat:add(self.forgetGate):add(nn.SelectTable(3))
   forget:add(concat)
   forget:add(nn.CMulTable())
   -- input = inputGate{input(t), output(t-1), cell(t-1)} * cellGate{input(t), output(t-1), cell(t-1)}
   local input = nn.Sequential()
   local concat2 = nn.ConcatTable()
   concat2:add(self.inputGate):add(self.cellGate)
   input:add(concat2)
   input:add(nn.CMulTable())
   -- cell(t) = forget + input
   local cell = nn.Sequential()
   local concat3 = nn.ConcatTable()
   concat3:add(forget):add(input)
   cell:add(concat3)
   cell:add(nn.CAddTable())
   self.cell = cell
   return cell
end   
   
function StepConvLSTM:buildOutputGate()
   self.outputGate = self:buildGate()
   return self.outputGate
end

-- cell(t) = cell{input, output(t-1), cell(t-1)}
-- output(t) = outputGate{input, output(t-1)}*tanh(cell(t))
-- output of Model is table : {output(t), cell(t)} 
function StepConvLSTM:buildModel_simVerson()
   -- Input is : {input(t), output(t-1), cell(t-1)}
   self.cell = self:buildcell()
   self.outputGate = self:buildOutputGate()
   -- assemble
   local concat = nn.ConcatTable()
   concat:add(nn.NarrowTable(1,2)):add(self.cell)
   local model = nn.Sequential()
   model:add(concat)
   -- output of concat is {{input(t), output(t-1)}, cell(t)}, 
   -- so flatten to {input(t), output(t-1), cell(t)}
   model:add(nn.FlattenTable())
   local cellAct = nn.Sequential()
   cellAct:add(nn.SelectTable(3))
   cellAct:add(nn.Tanh())
   local concat3 = nn.ConcatTable()
   concat3:add(self.outputGate):add(cellAct)
   local output = nn.Sequential()
   output:add(concat3)
   output:add(nn.CMulTable())
   -- we want the model to output : {output(t), cell(t)}
   local concat4 = nn.ConcatTable()
   concat4:add(output):add(nn.SelectTable(3))
   model:add(concat4)
   return model
end

--- KEEP FOR DEBUGGING ---

