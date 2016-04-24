require 'nn'
require 'rnn'
require 'FastConvLSTM'
require 'nngraph'
   nn.FastConvLSTM.usenngraph = true
typeTensor = torch.Tensor()
self = {}

  self.H = 50
  self.W = 50

   -- (inputSize, outputSize, rho, kc, km, stride, batchSize)

  print('---')
  self.kc =  3
  self.km =  3
  self.stride = stride or 1
  self.inputSize = inputSize or 5
  self.outputSize = outputSize or 8
  self.batchSize = batchSize or 3
  self.padc = math.floor(self.kc / 2) or 1
  self.padm = math.floor(self.km / 2) or 1
  self.rho = rho or 9999

---------
      self.i2g = nn.SpatialConvolution(self.inputSize, self.outputSize*4, self.kc, self.kc, self.stride, self.stride, self.padc, self.padc)

      self.o2g = nn.SpatialConvolution(self.outputSize, self.outputSize*4, self.km, self.km, self.stride, self.stride, self.padm, self.padm) 


----------

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
 
---------
local lstmSize_input = self.inputSize
local lstmSize_output = self.outputSize
local batchSize = self.batchSize
local nStep = self.rho - 1

print(batchSize, lstmSize_input, H, W)
typeTensor = torch.Tensor()
      local input = {}
      local gradOutput = {}

      step=1
         input[step] = torch.randn(batchSize, lstmSize_input, self.H, self.W):type(typeTensor:type())
      
      for step=2, 3 do
         input[step] = torch.randn(batchSize, lstmSize_output, self.H, self.W):type(typeTensor:type())
      end

      for step=1, 2 do

         gradOutput[step] = torch.randn(batchSize, lstmSize_output, self.H, self.W):type(typeTensor:type())
end

mlp:updateOutput(input)
mlp:updateGradInput(input, gradOutput)
mlp:accGradParameters(input, gradOutput)
graph.dot(mlp.fg, 'forwardlstm', 'forwardLSTM')

graph.dot(mlp.bg, 'backwardlstm', 'backwardLSTM')
-- graph.dot(mlp.fg, 'MLP')