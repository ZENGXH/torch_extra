-- test_FastConvLSTM.lua
require 'nn'
require 'rnn'
require 'FastConvLSTM'
   local rnntest = {}
   local mytester = torch.Tester()
function rnntest.lstm()
   local lstmSize = 1
   local batchSize = 4
   local nStep = 3
   -- (inputSize, outputSize, rho, i2g_kernel_size, o2g_kernel_size, stride)
   lstm1 = nn.FastConvLSTM(lstmSize, lstmSize, nStep, 3, 3, 1, batchSize) -- without nngraph
   local params1, gradParams1 = lstm1:getParameters()
   assert(torch.type(lstm1.recurrentModule) ~= 'nn.gModule')
   nn.FastConvLSTM.usenngraph = true

   lstm2 = nn.FastConvLSTM(lstmSize, lstmSize, nStep, 3, 3, 1, batchSize)  -- with nngraph
   nn.FastConvLSTM.usenngraph = false
   print(lstm2)


   local params2, gradParams2 = lstm2:getParameters()
   assert(torch.type(lstm2.recurrentModule) == 'nn.gModule')
   
   lstm2.i2g.weight:copy(lstm1.i2g.weight) 
   lstm2.i2g.bias:copy(lstm1.i2g.bias)
   lstm2.o2g.weight:copy(lstm1.o2g.weight)
   print('where fail:')
   print(params1[{{1, 10}}])
   print(params2[{{1, 10}}])
   print(params1:size())
   print(params2:size())
   mytester:assertTensorEq(params1, params2, 0.00000001, "FastConvLSTM nngraph params init err")
   -- mytester:assert(params1:size() == params2:size())
      -- , "FastConvLSTM nngraph params init err")

   lstm1:zeroGradParameters()
   lstm2:zeroGradParameters()
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastConvLSTM nngraph zeroGradParameters err")
   
   local seq1 = nn.Sequencer(lstm1)
   local seq2 = nn.Sequencer(lstm2)
   
   local input = {}
   local gradOutput = {}
   local H = 30
   local W = 30
   for step=1,nStep do
      input[step] = torch.randn(batchSize, lstmSize, H, W)
      gradOutput[step] = torch.randn(batchSize, lstmSize, H, W)
   end
   
   local rm1 = lstm1.recurrentModule
   local rm2 = lstm2.recurrentModule
   
   local input_ = {input[1], torch.randn(batchSize, lstmSize, H, W), torch.randn(batchSize, lstmSize, H, W)}
   local gradOutput_ = {gradOutput[1], torch.randn(batchSize, lstmSize, H, W)}
   print('input:', input_)

   local output1 = rm1:forward(input_)
   local output2 = rm2:forward(input_)
   rm1:zeroGradParameters()
   rm2:zeroGradParameters()
   local gradInput1 = rm1:backward(input_, gradOutput_)
   local gradInput2 = rm2:backward(input_, gradOutput_)
   
   mytester:assertTensorEq(output1[1], output2[1], 0.0000001, "FastConvLSTM.recurrentModule forward 1 error")
   mytester:assertTensorEq(output1[2], output2[2], 0.0000001, "FastConvLSTM.recurrentModule forward 2 error")
--   for i=1,3 do
--      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.0000001, "FastConvLSTM.recurrentModule backward err "..i)
--   end
   
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastConvLSTM.recurrenModule nngraph gradParams err")
   
   -- again, with sharedClone
   local rm3 = lstm1.recurrentModule:sharedClone()
   local rm4 = lstm2.recurrentModule:clone()
   
   local output1 = rm3:forward(input_)
   local output2 = rm4:forward(input_)
   local gradInput1 = rm3:backward(input_, gradOutput_)
   local gradInput2 = rm4:backward(input_, gradOutput_)
   
   mytester:assertTensorEq(output1[1], output2[1], 0.0000001, "FastConvLSTM.recurrentModule forward 1 error")
   mytester:assertTensorEq(output1[2], output2[2], 0.0000001, "FastConvLSTM.recurrentModule forward 2 error")
   for i=1,3 do
      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.0000001, "FastConvLSTM.recurrentModule backward err "..i)
   end
   
   local p1, gp1 = rm3:parameters()
   local p2, gp2 = rm4:parameters()
   
   for i=1,#p1 do
      mytester:assertTensorEq(gp1[i], gp2[i], 0.000001, "FastConvLSTM nngraph gradParam err "..i)
   end
   
   seq1:zeroGradParameters()
   seq2:zeroGradParameters()
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastConvLSTM nngraph zeroGradParameters err")
   mytester:assert(gradParams1:sum() == 0)
   
   local input_ = _.map(input, function(k, x) return x:clone() end)
   local gradOutput_ = _.map(gradOutput, function(k, x) return x:clone() end)
   
   -- forward/backward
   local output1 = seq1:forward(input)
   local gradInput1 = seq1:backward(input, gradOutput)
   local output2 = seq2:forward(input)
   local gradInput2 = seq2:backward(input, gradOutput)
   
   for i=1,#input do
      mytester:assertTensorEq(input[i], input_[i], 0.000001)
      mytester:assertTensorEq(gradOutput[i], gradOutput_[i], 0.000001)
   end
   
   for i=1,#output1 do
      mytester:assertTensorEq(output1[i], output2[i], 0.000001, "FastConvLSTM nngraph output error "..i)
      mytester:assertTensorEq(gradInput1[i], gradInput2[i], 0.000001, "FastConvLSTM nngraph gradInput error "..i)
   end
   
   local p1, gp1 = lstm2:parameters()
   local p2, gp2 = lstm2.sharedClones[2]:parameters()
   
   for i=1,#p1 do
      mytester:assertTensorEq(p1[i], p2[i], 0.000001, "FastConvLSTM nngraph param err "..i)
      mytester:assertTensorEq(gp1[i], gp2[i], 0.000001, "FastConvLSTM nngraph gradParam err "..i)
   end
   
   mytester:assertTensorEq(gradParams1, gradParams2, 0.000001, "FastConvLSTM nngraph gradParams err")
   --[[
   if benchmark and pcall(function() require 'cunn' end ) then
      local lstmSize = 128
      local batchSize = 50
      local nStep = 50
   
      local input = {}
      local gradOutput = {}
      for step=1,nStep do
         input[step] = torch.randn(batchSize, lstmSize, H, W):cuda()
         gradOutput[step] = torch.randn(batchSize, lstmSize, H, W):cuda()
      end
      
      nn.FastConvLSTM.usenngraph = false
      local lstm1 = nn.Sequencer(nn.FastConvLSTM(lstmSize)):cuda()
      nn.FastConvLSTM.usenngraph = true
      local lstm2 = nn.Sequencer(nn.FastConvLSTM(lstmSize)):cuda()
      nn.FastConvLSTM.usenngraph = false
      -- nn
      
      local output = lstm1:forward(input)
      cutorch.synchronize()
      local a = torch.Timer()
      for i=1,10 do
         lstm1:forward(input)
      end
      cutorch.synchronize()
      local nntime = a:time().real
      
      -- nngraph
      
      local output = lstm2:forward(input)
      cutorch.synchronize()
      local a = torch.Timer()
      for i=1,10 do
         lstm2:forward(input)
      end
      cutorch.synchronize()
      local nngraphtime = a:time().real
      
      print("Benchmark: nn vs nngraph time", nntime, nngraphtime)
   end
   ]]
end

   
   mytester:add(rnntest)
   mytester:run(tests)

