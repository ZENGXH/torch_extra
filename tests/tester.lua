
-- --prepareModeldir_Imagedir('test_StepConvLSTM')
dofile '../../hko/opts-hko.lua'
require 'nn'
require 'math'
dofile '../RecursiveSequential.lua'
dofile '../StepConvLSTM.lua'
dofile '../dataProvider.lua'

dofile '../utils.lua'
if not opt.onMac then
	require 'cunn'
	require 'cutorch'
	print('************ not on mac *********')
end
defaultType = 'nn.DoubleTensor'
runtest = {}
runtest2 = {}
-----------------------------------
function runtest2.hkotraning_recur()
	if not opt.onMac then
		require 'cunn'
		require 'cutorch'
	end
	local gpuid = selectFreeGpu()
	local data_path
	if not opt.onMac then
		data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
	else
		data_path = '../../../ConvLSTM/helper/'	
	end
	nn.StepConvLSTM.usenngraph = true
	trainDataProvider = getdataTensor_hko('train', data_path)
	assert(trainDataProvider[10000000])
	-- validDataProvider = getdataSeq_hko('valid', data_path)
		encoder_lstm0 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq , opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph

		encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq , opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph
		
		local bufferSize = 1
	    predictor_conv2 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
		                 		bufferSize, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
		                        opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph
	    -- predictor_conv3 and 2, output depth should be the same as encoder_lstm!
	    predictor_conv3 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		bufferSize, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
		                        opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph


	    encoder = nn.RecursiveSequential(opt.output_nSeq - 1):add(encoder_lstm0):add(encoder_lstm1)
	    -- print(encoder.modules[1].modules[1])

	    -- predictor = nn.SelfFeedSequencer( predictor_1 )
	    --[[predictor = nn.RecursiveSequential(opt.output_nSeq)
	    						:add(predictor_conv2)
	    						:add(predictor_conv3)
	    						:add(encoder_predictor)]]--
	    predictor = nn.Sequential():add(predictor_conv2):add(predictor_conv3)
		encoder_predictor = nn.SpatialConvolution(opt.nFiltersMemory[2]*2, opt.nFiltersMemory[1], 
	    										opt.decoderKernelSize, opt.decoderKernelSize, 1, 1, 
	    										math.floor(opt.decoderKernelSize / 2), 
	    										math.floor(opt.decoderKernelSize / 2))

		w_init(encoder_predictor)
	    print('number of parameters of repeatModel', encoder:getParameters():size(1) 
	    											+ predictor:getParameters():size(1)
	    											+ encoder_predictor:getParameters():size(1) )

	    print('encoder')
	  	print(encoder)
	  	print('predictor')
	  	print(predictor)
	  	

	  	-- local p_encoder_lstm0, g_encoder_lstm0 = encoder_lstm0:getParameters()
	  	-- local p_encoder_lstm1, g_encoder_lstm1 = encoder_lstm1:getParameters()
	  	-- assert(p_encoder_lstm1:size(1) + p_encoder_lstm0:size(1) == encoder:getParameters():size(1))
	  	-- local p_predictor_conv2, g_predictor_conv2 = predictor_conv2:getParameters()
	  	-- mytester:assertTensorEq(p_encoder_lstm0, parameters_encoder:narrow(1, 1,  p_encoder_lstm0:size(1)))
		-----------------------------------  building model done ----------------------------------


	
		epoch = epoch or 1  
		-- local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
	 	-- if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
		encoder:remember('both')
		encoder:training()
		predictor:remember('both') 
		predictor:training()

		local accErr = 0

	    --input_encoder = torch.Tensor((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    input_predictor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    target_predictor = torch.Tensor((opt.output_nSeq), opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)

	    criterion = nn.MSECriterion()

		for t =1, 30000 do
		    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
		    local tic = torch.tic()
		    
		    local iter = t
		    local trainData = trainDataProvider[t]
		    if t < 5 then
		    	checkMemory('start iter', gpuid)
		    end
		    input_encoder = trainData[1]
		    packBuffer(input_encoder)
		    target_predictor = trainData[2]
		    -- input_predictor = torch.Tensor(bufferSize * opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):fill(0)
		    input_predictor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):fill(0)

		    input_encoder_predictor = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[2]*2, opt.inputSizeH, opt.inputSizeW)
		    thre = torch.Tensor(opt.output_nSeq*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):fill(0.1)
		    mask = 10 * torch.gt(target_predictor,thre):double()

	    if not opt.onMac then
	    	encoder:cuda()
	    	predictor:cuda()
	    	encoder_predictor:cuda()
	    	criterion:cuda()

	    	input_encoder = input_encoder:cuda()
			input_predictor = input_predictor:cuda()
	    	target_predictor = target_predictor:cuda()
			input_encoder_predictor = input_encoder_predictor:cuda()
			-- thre = thre:cuda()
			mask = mask:cuda()
	    end


		    encoder_lstm1.module:zeroGradParameters()
		    encoder_lstm0.module:zeroGradParameters()

		    predictor_conv2.module:zeroGradParameters()
		    predictor_conv3.module:zeroGradParameters()
		    encoder_predictor:zeroGradParameters()

		    local output_encoder = encoder:forward(input_encoder)

		    predictor_conv2.initCell = encoder_lstm0.lastCell
		    predictor_conv3.initCell = encoder_lstm1.lastCell
		    predictor_conv2.initOutput = encoder_lstm0.lastOutput
		    predictor_conv3.initOutput = encoder_lstm1.lastOutput


			output_predictor = predictor:forward(input_predictor)
			packBuffer(output_predictor)

		    packBuffer(target_predictor)
  
		    --mytester:assert(output_predictor:size(1) == target_predictor:size(1))
		    --mytester:assert(output_predictor:size(2) == target_predictor:size(2))
		    mytester:assert(output_predictor:size(3) == target_predictor:size(3))
		    mytester:assert(output_predictor:size(4) == target_predictor:size(4))

		    -- ********* pack input **********
		    
		    unpackBuffer(encoder_lstm0.output)
		    unpackBuffer(encoder_lstm1.output)
		    unpackBuffer(predictor_conv2.output)
		    unpackBuffer(predictor_conv3.output)
		    --print(input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}]:size())
		    --print(encoder_lstm0.output:size())
		    input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}] = encoder_lstm0.output[{{1},{},{},{},{}}]
		    input_encoder_predictor[{{1},{},{1 + opt.nFiltersMemory[2], 2* opt.nFiltersMemory[2]},{},{}}] = encoder_lstm1.output[{{1},{},{},{},{}}]
		    input_encoder_predictor[{{2, opt.output_nSeq},{},{1, opt.nFiltersMemory[2]},{},{}}] = predictor_conv2.output
		    input_encoder_predictor[{{2, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2* opt.nFiltersMemory[2]},{},{}}] = predictor_conv3.output


		    packBuffer(encoder_lstm0.output)
		    packBuffer(encoder_lstm1.output)
		    packBuffer(predictor_conv2.output)
		    packBuffer(predictor_conv3.output)
		    -- *******************************
		    packBuffer(input_encoder_predictor)
		    prediction = encoder_predictor:forward(input_encoder_predictor)
		    -- unpackBuffer(input_encoder_predictor)
		    target_predictor = torch.add(target_predictor, mask)
		    accErr = criterion:forward(prediction, target_predictor)
		    gradOutput =  criterion:backward(prediction, target_predictor)

		    assert(gradOutput:size(1) == opt.output_nSeq*opt.batchSize)
		    assert(gradOutput:size(2) == target_predictor:size(2))
		    assert(gradOutput:size(3) == opt.inputSizeH)
		    assert(gradOutput)
		    --print(gradOutput:size())
		    --print(prediction:size())
		    grad_input_encoder_predictor = encoder_predictor:backward(input_encoder_predictor, gradOutput)

		    -- ************* unpack gradInput **********
		    unpackBuffer(grad_input_encoder_predictor)

		    local encoder_lstm0_grad_output = grad_input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}]
		    local encoder_lstm1_grad_output = grad_input_encoder_predictor[{{1},{},{1 + opt.nFiltersMemory[2], 2*opt.nFiltersMemory[2]},{},{}}]
		    local predictor_conv2_grad_output = grad_input_encoder_predictor[{{2, opt.output_nSeq},{},{1, opt.nFiltersMemory[2]},{},{}}]
		    local predictor_conv3_grad_output = grad_input_encoder_predictor[{{2, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2*	opt.nFiltersMemory[2]},{},{}}]

		    packBuffer(encoder_lstm0_grad_output)
		    packBuffer(encoder_lstm1_grad_output)
		    packBuffer(predictor_conv2_grad_output)
		    packBuffer(predictor_conv3_grad_output)
		    encoder_lstm0:maxBackWard(input_encoder, encoder_lstm0_grad_output)
		    encoder_lstm1:maxBackWard(encoder_lstm0.output, encoder_lstm1_grad_output)
		    predictor_conv2:backward(input_predictor, predictor_conv2_grad_output)
		    predictor_conv3:backward(predictor_conv2.output, predictor_conv3_grad_output)

		    -- *****************************************
		    --print('prediction mean:' , prediction:mean(), 'target mean ', target_predictor:mean(), 'predictor_conv2_grad_output: ',predictor_conv2_grad_output:mean())



		    -- gradInput = predictor:backward(input_predictor, grad_output_predictor)

		    -- output, accErr, gradInput = predictor:autoForwardAndBackward(input_predictor, target_predictor)


		    encoder_lstm0.gradPrevCell = predictor_conv2.gradPrevCell
		    encoder_lstm0:maxBackWard(input_encoder, predictor_conv2.lastGradPrevOutput)

		    encoder_lstm1.gradPrevCell = predictor_conv3.gradPrevCell
		    encoder_lstm1:maxBackWard(encoder_lstm0.output, predictor_conv3.lastGradPrevOutput)

		    encoder_lstm1:updateParameters(0.01)
		    encoder_lstm0:updateParameters(0.01)

		    predictor_conv2:updateParameters(0.01)
		    predictor_conv3:updateParameters(0.01)

		    mytester:assertTensorEq(predictor_conv2.initCell, encoder_lstm0.lastCell, 0.0001)

			print("\titer",t, "err:", accErr*10000)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	print('save')

		    	saveImage(output_encoder, 'output_encoder', t)
		    	unpackBuffer(prediction)
		    	unpackBuffer(target_predictor)
		    	unpackBuffer(encoder_lstm0.output)
		    	unpackBuffer(predictor_conv2_grad_output)
		    	unpackBuffer(predictor_conv3_grad_output)
		    	unpackBuffer(encoder_lstm0_grad_output)
		    	unpackBuffer(encoder_lstm1_grad_output)
		    	
		    	saveImage(grad_input_encoder_predictor,'grad_input_encoder_predictor', t)

		    	saveImage(encoder_lstm1_grad_output,'encoder_lstm1_grad_output', t)

		    	saveImage(predictor_conv3_grad_output, 'predictor_conv3_grad_output', t)

		    	saveImage(encoder_lstm0_grad_output,'encoder_lstm0_grad_output', t)
		    	saveImage(predictor_conv2_grad_output, 'predictor_conv2_grad_output', t)
		    	saveImage(encoder_lstm0.output, 'encoder_lstm0_output', t)
		    	saveImage(prediction, 'output', t)
		    	saveImage(target_predictor, 'target', t)

		    	packBuffer(encoder_lstm0.output)
		    	packBuffer(encoder_lstm0_grad_output)
		    	packBuffer(encoder_lstm1_grad_output)

		    	packBuffer(prediction)
		    	packBuffer(target_predictor)
		    	-- saveImage(gradInput, 'gradInput', t)
		    	-- { Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), }
			end
			local toc = torch.toc(tic)
			print('time used: ',toc)
		    if t < 5 then
		    	checkMemory('after iter', gpuid)
		    end
			collectgarbage()
		    if t < 5 then
		    	checkMemory('after collectgarbage', gpuid)
		    end
		end
end


-----------------------------------
function runtest.hkotraning()

	local gpuid = selectFreeGpu()
	local data_path
	if not opt.onMac then
		data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
	else
		data_path = '../../../ConvLSTM/helper/'	
	end
	nn.StepConvLSTM.usenngraph = true
	trainDataProvider = getdataTensor_hko('train', data_path)
	assert(trainDataProvider[1162])

	-- validDataProvider = getdataSeq_hko('valid', data_path)
		encoder_lstm0 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq , opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType, 1) -- without nngraph

		encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq , opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType, 1) -- without nngraph
		
		bufferSize = opt.output_nSeq - 1
	    predictor_conv2 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
		                 		bufferSize, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
		                        opt.inputSizeW, opt.inputSizeH, defaultType, 2) -- without nngraph
	    -- predictor_conv3 and 2, output depth should be the same as encoder_lstm!
	    predictor_conv3 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		bufferSize, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
		                        opt.inputSizeW, opt.inputSizeH, defaultType, 2) -- without nngraph


	    encoder = nn.Sequential():add(encoder_lstm0):add(encoder_lstm1)
	    -- print(encoder.modules[1].modules[1])

	    -- predictor = nn.SelfFeedSequencer( predictor_1 )
	    --[[predictor = nn.RecursiveSequential(opt.output_nSeq)
	    						:add(predictor_conv2)
	    						:add(predictor_conv3)
	    						:add(encoder_predictor)]]--
	    predictor = nn.Sequential():add(predictor_conv2):add(predictor_conv3)
		encoder_predictor = nn.SpatialConvolution(opt.nFiltersMemory[2]*2, opt.nFiltersMemory[1], 
	    										opt.decoderKernelSize, opt.decoderKernelSize, 1, 1, 
	    										math.floor(opt.decoderKernelSize / 2), 
	    										math.floor(opt.decoderKernelSize / 2))

		w_init(encoder_predictor)
	    print('number of parameters of repeatModel', encoder:getParameters():size(1) 
	    											+ predictor:getParameters():size(1)
	    											+ encoder_predictor:getParameters():size(1) )

	    print('encoder')
	  	print(encoder)
	  	print('predictor')
	  	print(predictor)
	  	print('encoder_predictor')
	  	print(encoder_predictor)

	  	-- local p_encoder_lstm0, g_encoder_lstm0 = encoder_lstm0:getParameters()
	  	-- local p_encoder_lstm1, g_encoder_lstm1 = encoder_lstm1:getParameters()
	  	-- assert(p_encoder_lstm1:size(1) + p_encoder_lstm0:size(1) == encoder:getParameters():size(1))
	  	-- local p_predictor_conv2, g_predictor_conv2 = predictor_conv2:getParameters()
	  	-- mytester:assertTensorEq(p_encoder_lstm0, parameters_encoder:narrow(1, 1,  p_encoder_lstm0:size(1)))
		-----------------------------------  building model done ----------------------------------


	
		epoch = epoch or 1  
		-- local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
	 	-- if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
		encoder:remember('both')
		encoder:training()
		predictor:remember('both') 
		predictor:training()

		local accErr = 0

	    --input_encoder = torch.Tensor((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    input_predictor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    target_predictor = torch.Tensor((opt.output_nSeq), opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)

	    criterion = nn.MSECriterion()

		for t =1, 30000 do
		    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
		    local tic = torch.tic()
		    
		    local iter = t

		    local trainData = trainDataProvider[t]
		    if t < 5 then
		    	checkMemory('start iter', gpuid)
		    end
		    input_encoder = trainData[1]
		    --input_encoder = trainData[1]:narrow(1, 1, opt.input_nSeq - 1) -- dimension1, from index 1, size = input_nSeq - 1
		    --input_predictor = trainData[1]:narrow(1, opt.input_nSeq, 1) 
		    -- input_encoder:resize((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW)
		    input_encoder = packBuffer(input_encoder)
		    mytester:assert(input_encoder:dim() == 4)

			--input_predictor:resize((1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW)
		    target_predictor = trainData[2]
		    -- print('target size', target_predictor:size())
		    input_predictor = torch.Tensor(bufferSize * opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):fill(0)
		    input_encoder_predictor = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.nFiltersMemory[2]*2, opt.inputSizeH, opt.inputSizeW)
		    thre = torch.Tensor(opt.output_nSeq*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):fill(0.5)
		    mask = 1 * torch.gt(target_predictor,thre):double()

	    if not opt.onMac then
	    	encoder:cuda()
	    	predictor:cuda()
	    	encoder_predictor:cuda()
	    	criterion:cuda()

	    	input_encoder = input_encoder:cuda()
			input_predictor = input_predictor:cuda()
	    	target_predictor = target_predictor:cuda()
			input_encoder_predictor = input_encoder_predictor:cuda()
			-- thre = thre:cuda()
			mask = mask:cuda()
	    end
--[[
    	    local parameters_encoder, gradParameters_encoder = encoder:getParameters()
		    assert(parameters_encoder, 'parameters_encoder empty') 
		    print('encoder parameters mean: ', parameters_encoder:mean())

    	    local parameters, gradParameters = predictor:getParameters()
		    print('predictor parameters mean: ', parameters:mean())

    	    parameters, gradParameters = encoder_predictor:getParameters()
		    print('encoder_predictor parameters mean: ', parameters:mean())
]]--

		    encoder_lstm1.module:zeroGradParameters()
		    encoder_lstm0.module:zeroGradParameters()

		    predictor_conv2.module:zeroGradParameters()
		    predictor_conv3.module:zeroGradParameters()
		    encoder_predictor:zeroGradParameters()

		    encoder:zeroGradParameters()
		    --[[print(encoder.module)
		    mlp = nn.Sequential():add(nn.Linear(2,3))
		    local pa, gra = mlp.modules[1]:getParameters()
		    gra:fill(2)
		    mlp:zeroGradParameters()
		    mytester:assert(gra:mean() == 0)
		    ]]--
		    -- encoder_lstm0:zeroGradParameters()
		   
		    -- mytester:assert(gradParameters_encoder:mean() == 0, 'mean is '..gradParameters_encoder:mean())
		    -- mytester:assert(gradParameters_encoder:mean() == 0, 'mean is '..gradParameters_encoder:mean())
		    --mytester:assert(g_encoder_lstm1:mean() == 0, 'mean is '..g_encoder_lstm1:mean())
		    
		    -- encoder:type(defaultType)
		    predictor:zeroGradParameters()

		    --mytester:assert(g_predictor_conv2:mean() == 0, 'mean is '..g_predictor_conv2:mean())
		    -- mytester:assert(g_encoder_lstm1:mean() == 0, 'mean is '..g_encoder_lstm1:mean())
		    
		    -- predictor:type(defaultType)

		    local output_encoder = encoder:forward(input_encoder)

		    predictor_conv2.initCell = encoder_lstm0.lastCell
		    predictor_conv3.initCell = encoder_lstm1.lastCell
		    predictor_conv2.initOutput = encoder_lstm0.lastOutput
		    predictor_conv3.initOutput = encoder_lstm1.lastOutput

		    --[[
		    local predictor_conv2_output = predictor_conv2:forward(input_predictor)
		    mytester:assert(output:size(1) == opt.output_nSeq*opt.batchSize)
		    mytester:assert(output:size(2) == opt.nFiltersMemory[2])
		    mytester:assert(output:size(3) == opt.inputSizeH)
		    -- print(predictor_conv3.module)
		    output = predictor_conv3:forward(output)
]]--
			--print('predictor')
			output_predictor = predictor:forward(input_predictor)
		    target_predictor = packBuffer(target_predictor)
  
		    --mytester:assert(output_predictor:size(1) == target_predictor:size(1))
		    --mytester:assert(output_predictor:size(2) == target_predictor:size(2))
		    mytester:assert(output_predictor:size(3) == target_predictor:size(3))
		    mytester:assert(output_predictor:size(4) == target_predictor:size(4))

		    -- ********* pack input **********
		    
		    encoder_lstm0.output = unpackBuffer(encoder_lstm0.output)
		    encoder_lstm1.output = unpackBuffer(encoder_lstm1.output)
		    predictor_conv2.output = unpackBuffer(predictor_conv2.output)
		    predictor_conv3.output = unpackBuffer(predictor_conv3.output)
		    --print(input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}]:size())
		    --print(encoder_lstm0.output:size())
		    input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}] = encoder_lstm0.output[{{1},{},{},{},{}}]
		    input_encoder_predictor[{{1},{},{1 + opt.nFiltersMemory[2], 2* opt.nFiltersMemory[2]},{},{}}] = encoder_lstm1.output[{{1},{},{},{},{}}]
		    input_encoder_predictor[{{2, opt.output_nSeq},{},{1, opt.nFiltersMemory[2]},{},{}}] = predictor_conv2.output
		    input_encoder_predictor[{{2, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2* opt.nFiltersMemory[2]},{},{}}] = predictor_conv3.output


		    encoder_lstm0.output = packBuffer(encoder_lstm0.output)
		    encoder_lstm1.output = packBuffer(encoder_lstm1.output)
		    predictor_conv2.output = packBuffer(predictor_conv2.output)
		    predictor_conv3.output = packBuffer(predictor_conv3.output)
		    -- *******************************
		    input_encoder_predictor = packBuffer(input_encoder_predictor)
		    prediction = encoder_predictor:forward(input_encoder_predictor)
		    prediction:clamp(0, 1)
		    -- unpackBuffer(input_encoder_predictor)
		    print('prediction max and min:', prediction:max(), prediction:min(), prediction:mean())
		    print('target max and min:', target_predictor:max(), target_predictor:min(), target_predictor:mean())

		    if opt.useMask then
		    	target_predictor = torch.add(target_predictor, mask)
			end		    	
		    accErr = criterion:forward(prediction, target_predictor)
		    gradOutput =  criterion:backward(prediction, target_predictor)

		    assert(gradOutput:size(1) == opt.output_nSeq*opt.batchSize)
		    assert(gradOutput:size(2) == target_predictor:size(2))
		    assert(gradOutput:size(3) == opt.inputSizeH)
		    assert(gradOutput)
		    --print(gradOutput:size())
		    --print(prediction:size())
		    grad_input_encoder_predictor = encoder_predictor:backward(input_encoder_predictor, gradOutput)
		    -- ************* unpack gradInput **********
		    grad_input_encoder_predictor = unpackBuffer(grad_input_encoder_predictor)

		    local encoder_lstm0_grad_output = grad_input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}]
		    local encoder_lstm1_grad_output = grad_input_encoder_predictor[{{1},{},{1 + opt.nFiltersMemory[2], 2*opt.nFiltersMemory[2]},{},{}}]
		    local predictor_conv2_grad_output = grad_input_encoder_predictor[{{2, opt.output_nSeq},{},{1, opt.nFiltersMemory[2]},{},{}}]
		    local predictor_conv3_grad_output = grad_input_encoder_predictor[{{2, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2*	opt.nFiltersMemory[2]},{},{}}]

		    encoder_lstm0_grad_output = packBuffer(encoder_lstm0_grad_output)
		    encoder_lstm1_grad_output = packBuffer(encoder_lstm1_grad_output)
		    predictor_conv2_grad_output = packBuffer(predictor_conv2_grad_output)
		    predictor_conv3_grad_output = packBuffer(predictor_conv3_grad_output)
		    encoder_lstm0:maxBackWard(input_encoder, encoder_lstm0_grad_output)
		    encoder_lstm1:maxBackWard(encoder_lstm0.output, encoder_lstm1_grad_output)

		    local gradInput_conv3 = predictor_conv3:backward(predictor_conv2.output, predictor_conv3_grad_output)
		    predictor_conv2_grad_output:add(gradInput_conv3)
		    predictor_conv2:backward(input_predictor, predictor_conv2_grad_output)

		    -- *****************************************
		    --print('prediction mean:' , prediction:mean(), 'target mean ', target_predictor:mean(), 'predictor_conv2_grad_output: ',predictor_conv2_grad_output:mean())



		    -- gradInput = predictor:backward(input_predictor, grad_output_predictor)

		    -- output, accErr, gradInput = predictor:autoForwardAndBackward(input_predictor, target_predictor)


		    encoder_lstm0.gradPrevCell = predictor_conv2.gradPrevCell
		    encoder_lstm0:maxBackWard(input_encoder, predictor_conv2.lastGradPrevOutput)

		    encoder_lstm1.gradPrevCell = predictor_conv3.gradPrevCell
		    encoder_lstm1:maxBackWard(encoder_lstm0.output, predictor_conv3.lastGradPrevOutput)
		    local lr = opt.lr or 1e-6
		    --[[
			momentumUpdateParameters(encoder_lstm1, lr)
			momentumUpdateParameters(encoder_lstm0, lr)
			momentumUpdateParameters(predictor_conv2, lr)
			momentumUpdateParameters(predictor_conv3, lr)
			momentumUpdateParameters(encoder_predictor, lr)
			]]--
			encoder_lstm1:updateParameters(lr)
			encoder_lstm0:updateParameters(lr)
			predictor_conv2:updateParameters(lr)
			predictor_conv3:updateParameters(lr)
			encoder_predictor:updateParameters(lr)

			-- weightVis(encoder_predictor, 'encoder_predictor_weight', t)

		    mytester:assertTensorEq(predictor_conv2.initCell, encoder_lstm0.lastCell, 0.0001)

			print("\titer",t, "err:", accErr*10000)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	print('save')
		    	saveImageSequence(packBuffer(grad_input_encoder_predictor),'seq_encoder_predictor_grad_input', t)
		    	saveImageDepth(packBuffer(grad_input_encoder_predictor),'dep_encoder_predictor_grad_input', t)
		    	saveImageSequence(encoder_lstm0_grad_output,'seq_lstm0_grad_output', t)
		    	saveImageDepth(encoder_lstm0_grad_output,'dep_lstm0_grad_output', t)
		    	saveImageSequence(encoder_lstm1_grad_output,'seq_lstm1_grad_output', t)
		    	saveImageDepth(encoder_lstm1_grad_output,'dep_lstm1_grad_output', t)
		    	saveImageSequence(predictor_conv2_grad_output,'seq_conv2_grad_output', t)
		    	saveImageDepth(predictor_conv2_grad_output,'dep_conv2_grad_output', t)
		    	saveImageSequence(predictor_conv3_grad_output,'seq_conv3_grad_output', t)
		    	saveImageDepth(predictor_conv3_grad_output,'dep_conv3_grad_output', t)
		    	saveImageSequence(input_encoder_predictor, 'dep_input_encoder_predictor', t)
		    	saveImageSequence(predictor_conv3.output,'seq_output_conv3', t)
		    	saveImageSequence(predictor_conv2.output,'seq_output_conv2', t)
		    	saveImageSequence(encoder_lstm0.output,'seq_output_lstm0', t)
		    	saveImageSequence(encoder_lstm1.output,'seq_output_lstm1', t)

		    	saveImageDepth(predictor_conv3.output,'dep_output_conv3', t)
		    	saveImageDepth(predictor_conv2.output,'dep_output_conv2', t)
		    	saveImageDepth(encoder_lstm0.output,'dep_output_lstm0', t)
		    	saveImageDepth(encoder_lstm1.output,'dep_output_lstm1', t)

		    	saveImageSequence(predictor_conv3.cells,'seq_cells_conv3', t)
		    	saveImageSequence(predictor_conv2.cells,'seq_cells_conv2', t)
		    	saveImageSequence(encoder_lstm0.cells,'seq_cells_lstm0', t)
		    	saveImageSequence(encoder_lstm1.cells,'seq_cells_lstm1', t)

		    	saveImageDepth(predictor_conv3.cells,'dep_cells_conv3', t)
		    	saveImageDepth(predictor_conv2.cells,'dep_cells_conv2', t)
		    	saveImageDepth(encoder_lstm0.cells,'dep_cells_lstm0', t)
		    	saveImageDepth(encoder_lstm1.cells,'dep_cells_lstm1', t)
		    	
		    
		    	saveImage(unpackBuffer(prediction), 'output', t)
		    	saveImage(unpackBuffer(target_predictor), 'target', t)

		    	--[[
		    	saveImage(output_encoder, 'output_encoder', t)
		    	prediction = unpackBuffer(prediction)
		    	target_predictor = unpackBuffer(target_predictor)
		    	-- encoder_lstm0.output = unpackBuffer(encoder_lstm0.output)
		    	-- predictor_conv2_grad_output = unpackBuffer(predictor_conv2_grad_output)
		    	-- predictor_conv3_grad_output = unpackBuffer(predictor_conv3_grad_output)
		    	-- encoder_lstm0_grad_output = unpackBuffer(encoder_lstm0_grad_output)
		    	-- encoder_lstm1_grad_output = unpackBuffer(encoder_lstm1_grad_output)
		    	-- saveImage(predictor_conv2.output, 'predictor_conv2.output', t, nil,nil,nil,true)
		    	saveImage(grad_input_encoder_predictor,'grad_input_encoder_predictor', t)

		    	saveImage(encoder_lstm0_grad_output:view(opt.input_nSeq, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.input_nSeq),'depth_encoder_lstm0_gradOutput', t)
		    	saveImage(encoder_lstm1_grad_output:view(opt.input_nSeq, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.input_nSeq),'depth_encoder_lstm1_gradOutput', t)

		    	saveImage(predictor_conv3.output:view(opt.output_nSeq - 1, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.output_nSeq - 1), 'depth_predictor_conv3_output', t)
		    	saveImage(predictor_conv2.output:view(opt.output_nSeq - 1, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.output_nSeq - 1), 'depth_predictor_conv3_output', t)
		    	saveImage(encoder_lstm1.output:view(opt.input_nSeq, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.input_nSeq), 'depth_encoder_lstm1_output', t)
		    	saveImage(encoder_lstm0.output:view(opt.input_nSeq, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.input_nSeq), 'depth_encoder_lstm0_output', t)

		    	saveImage(predictor_conv2_grad_output:view(opt.input_nSeq, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.input_nSeq),'depth_predictor_conv2_gradOutput', t)
		    	saveImage(predictor_conv3_grad_output:view(opt.input_nSeq, opt.batchSize, 
		    									opt.nFiltersMemory[2]*opt.inputSizeH, opt.inputSizeW):select(1, opt.input_nSeq), 'depth_predictor_conv3_gradOutput', t)


		    	saveImage(predictor_conv3.cells, 'predictor_conv3_cell', t, nil,nil,nil,true)

		    	-- saveImage(predictor_conv2.cells, 'predictor_conv3_cell', t, nil,nil,nil,true)

		    	saveImage(encoder_lstm1.output, 'encoder_lstm1.output', t, nil,nil,nil,true)

		    	saveImage(encoder_lstm0.output, 'encoder_lstm0_output', t)
		    	saveImage(prediction, 'output', t)
		    	saveImage(target_predictor, 'target', t)
		    	saveImage(input_encoder_predictor, 'input_encoder_predictor',t, nil,nil,nil,true)
		    	-- encoder_lstm0.output = packBuffer(encoder_lstm0.output)
		    	-- encoder_lstm0_grad_output = packBuffer(encoder_lstm0_grad_output)
		    	-- encoder_lstm1_grad_output = packBuffer(encoder_lstm1_grad_output)

		    	prediction = packBuffer(prediction)
		    	target_predictor = packBuffer(target_predictor)
				]]
		    	-- saveImage(gradInput, 'gradInput', t)
		    	-- { Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), }
			end
			local toc = torch.toc(tic)
			print('time used: ',toc)
		    if t < 5 then
		    	checkMemory('after iter', gpuid)
		    end
			collectgarbage()
		    if t < 5 then
		    	checkMemory('after collectgarbage', gpuid)
		    end
		end
end

function runtest.RecursiveSequential()

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
	-- print("outputsize: \n", output:size())
	-- print("gradInput size \n", gradInput:size())
end

function runtest.StepConvLSTM()
	
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
	for i = 1, 3 do
		output = net:forward(input)
		output = net:unpackBuffer(output)
		mytester:assertTensorEq(output, net.debugBuffer)
	end
	gradOutput = torch.randn(bufferStep * batchSize, outputSize, height, width)

	gradInput = net:backward(input, gradOutput)

	gradOutput_sub = torch.randn((2) * batchSize, outputSize, height, width)
	print('maxibackward')
	net:maxBackWard(input, gradOutput_sub)
end


function runtest.momentumUpdateParameters()
	local net = nn.Linear(2, 3)
	local parameters, gradParameters = net:getParameters()
	net:zeroGradParameters()
	local ori = parameters:clone()
	local input = torch.randn(2)
	if not opt.onMac then
		input = input:cuda()
		net:cuda()
		ori = ori:cuda()
	end

	local output = net:forward(input)
	local gradOutput = torch.randn(3)
	if not opt.onMac then
		output = output:cuda()
		gradOutput = gradOutput:cuda()
	end

	net:backward(input, gradOutput)


	momentumUpdateParameters(net)

	local parameters2, gradParameters = net:getParameters()
	mytester:assertTensorNe(parameters2, ori)
	momentumUpdateParameters(net)

	local parameters2, gradParameters = net:getParameters()
	mytester:assertTensorNe(parameters2, ori)
end

function runtest.weightVis()
	local net = nn.SpatialConvolution(2, 3, 3, 3, 1, 1, 1, 1)
	local parameters, gradParameters = net:getParameters()
	net:zeroGradParameters()
	local ori = parameters:clone()
	local input = torch.randn(2,10,10)
	local output = net:forward(input)
	local gradOutput = torch.randn(3, 10,10)
	net:backward(input, gradOutput)
	print('save image test')
	weightVis(net, 'test', 0, './')
end
mytester = torch.Tester()
mytester:add(runtest)
-- math.randomseed(os.time())
mytester:run()
