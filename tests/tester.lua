
-- --prepareModeldir_Imagedir('test_StepConvLSTM')
dofile '../../hko/opts-hko.lua'
require 'nn'
require 'math'
dofile '../RecursiveSequential.lua'
dofile '../StepConvLSTM.lua'
dofile '../dataProvider.lua'
dofile '../HadamardMul.lua'
dofile '../utils.lua'
if not opt.onMac then
	require 'cunn'
	require 'cutorch'
	print('************ not on mac *********')
end
defaultType = 'nn.DoubleTensor'
runtest = {}
runtest2 = {}
torch.manualSeed(999)
	local epochSaveDir = './data/'..curtime..'tempinit_copy/'
------------------------------
function runtest2.hkotraning_inputPredictor()

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
		                 		opt.input_nSeq - 1, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType, 1) -- without nngraph

		encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq - 1, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType, 1) -- without nngraph
		
		bufferSize = opt.output_nSeq
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
	    
		    local p1, g = encoder:getParameters()
		    print('encoder weight: ', p1:mean())
		--    p:uniform(-0.08,0.08)
		    local p, g = predictor:getParameters()
		--    p:uniform(-0.08,0.08)
			p:copy(p1):mul(1/2)
		    print('predictior weight: ', p:mean())
			print('lstm weight init: p:copy(p1):mul(1/2)')
		--    local p, g = encoder_predictor:getParameters()
		--    p:uniform(-0.08,0.08)
		--    print('encoder_predictor weight: ', p:mean())

		for t =1, 32000 do
			if t == 10000 then
				opt.lr = opt.lr * 0.1
			end
			if t == 20000 then
				opt.lr = opt.lr * 0.1
			end

		    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
		    local tic = torch.tic()
		    
		    local iter = t

		    local trainData = trainDataProvider[t]
		    if t < 5 then
		    	checkMemory('start iter', gpuid)
		    end
		    local p, g = encoder:getParameters()
		    print('encoder weight: ', p:mean())
		    --p:uniform(-0.08,0.08)
		    local p, g = predictor:getParameters()
		    --p:uniform(-0.08,0.08)
		    print('predictior weight: ', p:mean())
		    local p, g = encoder_predictor:getParameters()
		    --p:uniform(-0.08,0.08)
		    print('encoder_predictor weight: ', p:mean())

		    -- input_encoder = trainData[1]
		    input_encoder = trainData[1]:narrow(1, 1, opt.input_nSeq - 1) -- dimension1, from index 1, size = input_nSeq - 1
		    -- input_encoder:resize((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW)
		    input_encoder = packBuffer(input_encoder)
		    mytester:assert(input_encoder:dim() == 4)

			--input_predictor:resize((1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW)
		    target_predictor = trainData[2]
		    -- print('target size', target_predictor:size())
		    input_predictor = torch.Tensor(bufferSize, opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):fill(0)
		    input_predictor[1] = trainData[1]:narrow(1, opt.input_nSeq, 1) 
		    input_predictor = packBuffer(input_predictor)

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

		    encoder_lstm1.module:zeroGradParameters()
		    encoder_lstm0.module:zeroGradParameters()

		    predictor_conv2.module:zeroGradParameters()
		    predictor_conv3.module:zeroGradParameters()
		    encoder_predictor:zeroGradParameters()

		    encoder:zeroGradParameters()

		    predictor:zeroGradParameters()

		    local output_encoder = encoder:forward(input_encoder)

		    predictor_conv2.initCell = encoder_lstm0.lastCell
		    predictor_conv3.initCell = encoder_lstm1.lastCell
		    predictor_conv2.initOutput = encoder_lstm0.lastOutput
		    predictor_conv3.initOutput = encoder_lstm1.lastOutput


			output_predictor = predictor:forward(input_predictor)
		    target_predictor = packBuffer(target_predictor)
  
		    mytester:assert(output_predictor:size(3) == target_predictor:size(3))
		    mytester:assert(output_predictor:size(4) == target_predictor:size(4))

		    -- ********* pack input **********
		    
		    encoder_lstm0.output = unpackBuffer(encoder_lstm0.output)
		    encoder_lstm1.output = unpackBuffer(encoder_lstm1.output)
		    predictor_conv2.output = unpackBuffer(predictor_conv2.output)
		    predictor_conv3.output = unpackBuffer(predictor_conv3.output)
		    --print(input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}]:size())
		    --print(encoder_lstm0.output:size())
		    --input_encoder_predictor[{{1},{},{1, opt.nFiltersMemory[2]},{},{}}] = encoder_lstm0.output[{{1},{},{},{},{}}]
		    --input_encoder_predictor[{{1},{},{1 + opt.nFiltersMemory[2], 2* opt.nFiltersMemory[2]},{},{}}] = encoder_lstm1.output[{{1},{},{},{},{}}]
		    input_encoder_predictor[{{1, opt.output_nSeq},{},{1, opt.nFiltersMemory[2]},{},{}}] = predictor_conv2.output
		    input_encoder_predictor[{{1, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2* opt.nFiltersMemory[2]},{},{}}] = predictor_conv3.output


		    encoder_lstm0.output = packBuffer(encoder_lstm0.output)
		    encoder_lstm1.output = packBuffer(encoder_lstm1.output)
		    predictor_conv2.output = packBuffer(predictor_conv2.output)
		    predictor_conv3.output = packBuffer(predictor_conv3.output)
		    -- *******************************
		    input_encoder_predictor = packBuffer(input_encoder_predictor)
		    prediction = encoder_predictor:forward(input_encoder_predictor)
		    -- prediction:clamp(0, 1)
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
		    local predictor_conv2_grad_output = grad_input_encoder_predictor[{{1, opt.output_nSeq},{},{1, opt.nFiltersMemory[2]},{},{}}]
		    local predictor_conv3_grad_output = grad_input_encoder_predictor[{{1, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2*opt.nFiltersMemory[2]},{},{}}]

		    encoder_lstm0_grad_output = packBuffer(encoder_lstm0_grad_output)
		    encoder_lstm1_grad_output = packBuffer(encoder_lstm1_grad_output)
		    predictor_conv2_grad_output = packBuffer(predictor_conv2_grad_output)
		    predictor_conv3_grad_output = packBuffer(predictor_conv3_grad_output)
		    -- encoder_lstm0:maxBackWard(input_encoder, encoder_lstm0_grad_output)
		    -- encoder_lstm1:maxBackWard(encoder_lstm0.output, encoder_lstm1_grad_output)

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
			momentumUpdateParameters(encoder_lstm1.module, lr)
			momentumUpdateParameters(encoder_lstm0.module, lr)
			momentumUpdateParameters(predictor_conv2.module, lr)
			momentumUpdateParameters(predictor_conv3.module, lr)
			momentumUpdateParameters(encoder_predictor, lr)
		
			encoder_lstm1.module:updateParameters(lr*10)
			encoder_lstm0.module:updateParameters(lr*10)
			predictor_conv2.module:updateParameters(lr)
			predictor_conv3.module:updateParameters(lr)
			encoder_predictor:updateParameters(lr)
]]--
			-- weightVis(encoder_predictor, 'encoder_predictor_weight', t)

		    mytester:assertTensorEq(predictor_conv2.initCell, encoder_lstm0.lastCell, 0.0001)

			print("\titer",t, "err:", accErr*10000)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	print('save')
		    	saveImageSequence(packBuffer(grad_input_encoder_predictor),'seq_encoder_predictor_grad_input', t, epochSaveDir)
		    	saveImageDepth(packBuffer(grad_input_encoder_predictor),'dep_encoder_predictor_grad_input', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm0_grad_output,'seq_lstm0_grad_output', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm0_grad_output,'dep_lstm0_grad_output', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm1_grad_output,'seq_lstm1_grad_output', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm1_grad_output,'dep_lstm1_grad_output', t, epochSaveDir)
		    	saveImageSequence(predictor_conv2_grad_output,'seq_conv2_grad_output', t, epochSaveDir)
		    	saveImageDepth(predictor_conv2_grad_output,'dep_conv2_grad_output', t, epochSaveDir)
		    	saveImageSequence(predictor_conv3_grad_output,'seq_conv3_grad_output', t, epochSaveDir)
		    	saveImageDepth(predictor_conv3_grad_output,'dep_conv3_grad_output', t, epochSaveDir)
		    	local allin = unpackBuffer(input_encoder_predictor)

		    	allin = makeContiguous(allin:select(2, 1))
		    	saveImageAll(allin, 'allin', t, epochSaveDir)
		    	-- allin = allin:resize(allin:size(1) * allin:size(2), allin:size(2)*allin:size(4))
		    	-- saveImageSequence(input_encoder_predictor, 'dep_input_encoder_predictor', t)
				-- print('sizei of all in ', allin:size())
		    	-- saveImage(allin, 'allin', t)

		    	saveImageAll(unpackBuffer(predictor_conv3.output):select(2,1), 'all_output_conv3', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv2.output):select(2,1), 'all_output_conv2', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(encoder_lstm0.output):select(2,1), 'all_output_lstm0', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(encoder_lstm1.output):select(2,1), 'all_output_lstm1', t, epochSaveDir)

		    	saveImageAll(unpackBuffer(prediction):select(2,1), 'all_prediction', t, epochSaveDir)


		    	saveImageAll(unpackBuffer(target_predictor):select(2,1), 'all_target', t, epochSaveDir)


		    	saveImageSequence(predictor_conv3.output,'seq_output_conv3', t, epochSaveDir)
		    	saveImageSequence(predictor_conv2.output,'seq_output_conv2', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm0.output,'seq_output_lstm0', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm1.output,'seq_output_lstm1', t, epochSaveDir)

		    	saveImageDepth(predictor_conv3.output,'dep_output_conv3', t, epochSaveDir)
		    	saveImageDepth(predictor_conv2.output,'dep_output_conv2', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm0.output,'dep_output_lstm0', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm1.output,'dep_output_lstm1', t, epochSaveDir)

		    	saveImageSequence(predictor_conv3.cells,'seq_cells_conv3', t, epochSaveDir)
		    	saveImageSequence(predictor_conv2.cells,'seq_cells_conv2', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm0.cells,'seq_cells_lstm0', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm1.cells,'seq_cells_lstm1', t, epochSaveDir)

		    	saveImageDepth(predictor_conv3.cells,'dep_cells_conv3', t, epochSaveDir)
		    	saveImageDepth(predictor_conv2.cells,'dep_cells_conv2', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm0.cells,'dep_cells_lstm0', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm1.cells,'dep_cells_lstm1', t, epochSaveDir)
		    	
		    	saveImage(unpackBuffer(prediction/prediction:max()), 'output2', t, epochSaveDir)

		    	saveImage(unpackBuffer(prediction), 'output', t, epochSaveDir)
		    	saveImage(unpackBuffer(target_predictor), 'target', t, epochSaveDir)

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


----------------------------------
function runtest2.hkotraning()

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

	encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[1],  -- 5, 15?
	                 		opt.input_nSeq , opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, 
	                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType, 1) -- without nngraph
	
	bufferSize = 1 -- opt.output_nSeq 
    predictor_conv2 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		bufferSize, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
	                        opt.inputSizeW, opt.inputSizeH, defaultType, 2) -- without nngraph
    -- predictor_conv3 and 2, output depth should be the same as encoder_lstm!
    predictor_conv3 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[1],  -- 5, 15?
	                 		bufferSize, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
	                        opt.inputSizeW, opt.inputSizeH, defaultType, 2) -- without nngraph


    encoder = nn.Sequential():add(encoder_lstm0):add(encoder_lstm1)
    -- print(encoder.modules[1].modules[1])

    -- predictor = nn.SelfFeedSequencer( predictor_1 )
    predictor = nn.RecursiveSequential(opt.output_nSeq)
    						:add(predictor_conv2)
    						:add(predictor_conv3)
    						--:add(encoder_predictor)]]--
    --predictor = nn.Sequential():add(predictor_conv2):add(predictor_conv3)
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

	    encoder_lstm1.module:zeroGradParameters()
	    encoder_lstm0.module:zeroGradParameters()

	    predictor_conv2.module:zeroGradParameters()
	    predictor_conv3.module:zeroGradParameters()
	    encoder_predictor:zeroGradParameters()

	    encoder:zeroGradParameters()

	    predictor:zeroGradParameters()


	    local output_encoder = encoder:forward(input_encoder)

	    predictor_conv2.initCell = encoder_lstm0.lastCell
	    predictor_conv3.initCell = encoder_lstm1.lastCell
	    predictor_conv2.initOutput = encoder_lstm0.lastOutput
	    predictor_conv3.initOutput = encoder_lstm1.lastOutput
prediction, accErr, gradOutput = predictor:autoForwardAndBackward(input_predictor, target_predictor)
-- print(gradOutput:size()) -- 3,1,100,100 (batch, depth, h, w)
	    assert(gradOutput:size(1) == opt.batchSize)
	    assert(gradOutput:size(3) == opt.inputSizeH)
	    assert(gradOutput:size(4) == opt.inputSizeW)
	    -- target_predictor = packBuffer(target_predictor)

	    -- ********* pack input **********
	    
	    encoder_lstm0.output = unpackBuffer(encoder_lstm0.output)
	    encoder_lstm1.output = unpackBuffer(encoder_lstm1.output)
	    predictor_conv2.output = unpackBuffer(predictor_conv2.output)
	    predictor_conv3.output = unpackBuffer(predictor_conv3.output)

	    encoder_lstm0.output = packBuffer(encoder_lstm0.output)
	    encoder_lstm1.output = packBuffer(encoder_lstm1.output)
	    predictor_conv2.output = packBuffer(predictor_conv2.output)
	    predictor_conv3.output = packBuffer(predictor_conv3.output)
	    -- *******************************

	    print('prediction max and min:', prediction:max(), prediction:min(), prediction:mean())
	    print('target max and min:', target_predictor:max(), target_predictor:min(), target_predictor:mean())

	    if opt.useMask then
	    	target_predictor = torch.add(target_predictor, mask)
		end		    	
		-- print(target_predictor:size())
		-- print(prediction:size())
	    -- accErr = criterion:forward(prediction, target_predictor)
	    -- gradOutput =  criterion:backward(prediction, target_predictor)

	    -- assert(gradOutput:size(1) == opt.output_nSeq*opt.batchSize)

	    -- assert(gradOutput:size(2) == target_predictor:size(2))
	    -- assert(gradOutput:size(3) == opt.inputSizeH)
	    -- assert(gradOutput)
	    --print(gradOutput:size())
	    --print(prediction:size())
	    -- gradInput_predictor = predictor:backward(input_predictor, gradOutput)
		--	grad_input_encoder_predictor = encoder_predictor:backward(input_encoder_predictor, gradOutput)

	    encoder_lstm0.gradPrevCell = predictor_conv2.gradPrevCell
	    encoder_lstm0:maxBackWard(input_encoder, predictor_conv2.lastGradPrevOutput)

	    encoder_lstm1.gradPrevCell = predictor_conv3.gradPrevCell
	    encoder_lstm1:maxBackWard(encoder_lstm0.output, predictor_conv3.lastGradPrevOutput)
	    local lr = opt.lr or 1e-6
	    
		momentumUpdateParameters(encoder_lstm1, lr)
		momentumUpdateParameters(encoder_lstm0, lr)
		momentumUpdateParameters(predictor_conv2, lr)
		momentumUpdateParameters(predictor_conv3, lr)
		momentumUpdateParameters(encoder_predictor, lr)
		--[[		
				encoder_lstm1:updateParameters(lr)
				encoder_lstm0:updateParameters(lr)
				predictor_conv2:updateParameters(lr)
				predictor_conv3:updateParameters(lr)
				encoder_predictor:updateParameters(lr)
		]]--
				-- weightVis(encoder_predictor, 'encoder_predictor_weight', t)

		    mytester:assertTensorEq(predictor_conv2.initCell, encoder_lstm0.lastCell, 0.0001)

			print("\titer",t, "err:", accErr*10000)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	print('save')

		    	saveImageAll(unpackBuffer(predictor_conv3.output):select(2,1), 'all_output_conv3', t)
		    	saveImageAll(unpackBuffer(predictor_conv2.output):select(2,1), 'all_output_conv2', t)
		    	saveImageAll(unpackBuffer(encoder_lstm0.output):select(2,1), 'all_output_lstm0', t)
		    	saveImageAll(unpackBuffer(encoder_lstm1.output):select(2,1), 'all_output_lstm1', t)

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
		    	saveImage(prediction, 'output', t)
		    	saveImage(target_predictor, 'target', t)

--		    	saveImage(unpackBuffer(prediction), 'output', t)
--		    	saveImage(unpackBuffer(target_predictor), 'target', t)
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
function runtest2.hkotraning_noconv()

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

	encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[1],  -- 5, 15?
	                 		opt.input_nSeq , opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, 
	                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType, 1) -- without nngraph
	
	bufferSize = opt.output_nSeq 
    predictor_conv2 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		bufferSize, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
	                        opt.inputSizeW, opt.inputSizeH, defaultType, 2) -- without nngraph
    -- predictor_conv3 and 2, output depth should be the same as encoder_lstm!
    predictor_conv3 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[1],  -- 5, 15?
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

	    encoder_lstm1.module:zeroGradParameters()
	    encoder_lstm0.module:zeroGradParameters()

	    predictor_conv2.module:zeroGradParameters()
	    predictor_conv3.module:zeroGradParameters()
	    encoder_predictor:zeroGradParameters()

	    encoder:zeroGradParameters()

	    predictor:zeroGradParameters()


	    local output_encoder = encoder:forward(input_encoder)

	    predictor_conv2.initCell = encoder_lstm0.lastCell
	    predictor_conv3.initCell = encoder_lstm1.lastCell
	    predictor_conv2.initOutput = encoder_lstm0.lastOutput
	    predictor_conv3.initOutput = encoder_lstm1.lastOutput

		prediction = predictor:forward(input_predictor)
	    target_predictor = packBuffer(target_predictor)

	    -- ********* pack input **********
	    
	    encoder_lstm0.output = unpackBuffer(encoder_lstm0.output)
	    encoder_lstm1.output = unpackBuffer(encoder_lstm1.output)
	    predictor_conv2.output = unpackBuffer(predictor_conv2.output)
	    predictor_conv3.output = unpackBuffer(predictor_conv3.output)

	    encoder_lstm0.output = packBuffer(encoder_lstm0.output)
	    encoder_lstm1.output = packBuffer(encoder_lstm1.output)
	    predictor_conv2.output = packBuffer(predictor_conv2.output)
	    predictor_conv3.output = packBuffer(predictor_conv3.output)
	    -- *******************************

	    print('prediction max and min:', prediction:max(), prediction:min(), prediction:mean())
	    print('target max and min:', target_predictor:max(), target_predictor:min(), target_predictor:mean())

	    if opt.useMask then
	    	target_predictor = torch.add(target_predictor, mask)
		end		    	
		-- print(target_predictor:size())
		-- print(prediction:size())
	    accErr = criterion:forward(prediction, target_predictor)
	    gradOutput =  criterion:backward(prediction, target_predictor)

	    assert(gradOutput:size(1) == opt.output_nSeq*opt.batchSize)
	    assert(gradOutput:size(2) == target_predictor:size(2))
	    assert(gradOutput:size(3) == opt.inputSizeH)
	    assert(gradOutput)
	    --print(gradOutput:size())
	    --print(prediction:size())
	    gradInput_predictor = predictor:backward(input_predictor, gradOutput)
		--	grad_input_encoder_predictor = encoder_predictor:backward(input_encoder_predictor, gradOutput)

	    encoder_lstm0.gradPrevCell = predictor_conv2.gradPrevCell
	    encoder_lstm0:maxBackWard(input_encoder, predictor_conv2.lastGradPrevOutput)

	    encoder_lstm1.gradPrevCell = predictor_conv3.gradPrevCell
	    encoder_lstm1:maxBackWard(encoder_lstm0.output, predictor_conv3.lastGradPrevOutput)
	    local lr = opt.lr or 1e-6
	    
		momentumUpdateParameters(encoder_lstm1, lr)
		momentumUpdateParameters(encoder_lstm0, lr)
		momentumUpdateParameters(predictor_conv2, lr)
		momentumUpdateParameters(predictor_conv3, lr)
		momentumUpdateParameters(encoder_predictor, lr)
		--[[		
				encoder_lstm1:updateParameters(lr)
				encoder_lstm0:updateParameters(lr)
				predictor_conv2:updateParameters(lr)
				predictor_conv3:updateParameters(lr)
				encoder_predictor:updateParameters(lr)
		]]--
				-- weightVis(encoder_predictor, 'encoder_predictor_weight', t)

		    mytester:assertTensorEq(predictor_conv2.initCell, encoder_lstm0.lastCell, 0.0001)

			print("\titer",t, "err:", accErr*10000)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	print('save')

		    	saveImageAll(unpackBuffer(predictor_conv3.output):select(2,1), 'all_output_conv3', t)
		    	saveImageAll(unpackBuffer(predictor_conv2.output):select(2,1), 'all_output_conv2', t)
		    	saveImageAll(unpackBuffer(encoder_lstm0.output):select(2,1), 'all_output_lstm0', t)
		    	saveImageAll(unpackBuffer(encoder_lstm1.output):select(2,1), 'all_output_lstm1', t)
				--[[
    			local dd = image.toDisplayTensor{input=unpackBuffer(prediction):select(2,1),
                           padding=0.4,
                           nrow= 1,
                           }
 				local dd = image.toDisplayTensor{input=unpackBuffer(target_predictor):select(2,1),
                           padding=0.4,
                           nrow= 1,
                           }
                ]]--
		    	saveImageAll(unpackBuffer(prediction):select(2,1), 'all_prediction', t)


		    	saveImageAll(unpackBuffer(target_predictor):select(2,1), 'all_target', t)

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
function runtest2.hkotraning_ori()

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
	    
		    local p1, g = encoder:getParameters()
		    print('encoder weight: ', p1:mean())
		--    p:uniform(-0.08,0.08)
		    local p, g = predictor:getParameters()
		--    p:uniform(-0.08,0.08)
			-- p:copy(p1)
		    print('predictior weight: ', p:mean())
			print('lstm weight init: p:copy(p1):mul(1/2)')
		--    local p, g = encoder_predictor:getParameters()
		--    p:uniform(-0.08,0.08)
		--    print('encoder_predictor weight: ', p:mean())

		for t =1, 32000 do
			if t == 10000 then
				opt.lr = opt.lr * 0.1
			end
			if t == 20000 then
				opt.lr = opt.lr * 0.1
			end

		    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
		    local tic = torch.tic()
		    
		    local iter = t

		    local trainData = trainDataProvider[t]
		    if t < 5 then
		    	checkMemory('start iter', gpuid)
		    end
		    local p, g = encoder:getParameters()
		    print('encoder weight: ', p:mean())
		    --p:uniform(-0.08,0.08)
		    local p, g = predictor:getParameters()
		    --p:uniform(-0.08,0.08)
		    print('predictior weight: ', p:mean())
		    local p, g = encoder_predictor:getParameters()
		    --p:uniform(-0.08,0.08)
		    print('encoder_predictor weight: ', p:mean())

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

		    encoder_lstm1.module:zeroGradParameters()
		    encoder_lstm0.module:zeroGradParameters()

		    predictor_conv2.module:zeroGradParameters()
		    predictor_conv3.module:zeroGradParameters()
		    encoder_predictor:zeroGradParameters()

		    encoder:zeroGradParameters()

		    predictor:zeroGradParameters()

		    local output_encoder = encoder:forward(input_encoder)

		    predictor_conv2.initCell = encoder_lstm0.lastCell
		    predictor_conv3.initCell = encoder_lstm1.lastCell
		    predictor_conv2.initOutput = encoder_lstm0.lastOutput
		    predictor_conv3.initOutput = encoder_lstm1.lastOutput


			output_predictor = predictor:forward(input_predictor)
		    target_predictor = packBuffer(target_predictor)
  
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
		    -- prediction:clamp(0, 1)
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
		    local predictor_conv3_grad_output = grad_input_encoder_predictor[{{2, opt.output_nSeq},{},{1 + opt.nFiltersMemory[2], 2*opt.nFiltersMemory[2]},{},{}}]

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
			momentumUpdateParameters(encoder_lstm1.module, lr)
			momentumUpdateParameters(encoder_lstm0.module, lr)
			momentumUpdateParameters(predictor_conv2.module, lr)
			momentumUpdateParameters(predictor_conv3.module, lr)
			momentumUpdateParameters(encoder_predictor, lr)
	]]--	
			encoder_lstm1.module:updateParameters(lr*10)
			encoder_lstm0.module:updateParameters(lr*10)
			predictor_conv2.module:updateParameters(lr)
			predictor_conv3.module:updateParameters(lr)
			encoder_predictor:updateParameters(lr)

			-- weightVis(encoder_predictor, 'encoder_predictor_weight', t)

		    mytester:assertTensorEq(predictor_conv2.initCell, encoder_lstm0.lastCell, 0.0001)

			print("\titer",t, "err:", accErr*10000)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	print('save')
		    	--[[
		    	saveImageSequence(packBuffer(grad_input_encoder_predictor),'seq_encoder_predictor_grad_input', t, epochSaveDir)
		    	saveImageDepth(packBuffer(grad_input_encoder_predictor),'dep_encoder_predictor_grad_input', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm0_grad_output,'seq_lstm0_grad_output', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm0_grad_output,'dep_lstm0_grad_output', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm1_grad_output,'seq_lstm1_grad_output', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm1_grad_output,'dep_lstm1_grad_output', t, epochSaveDir)
		    	saveImageSequence(predictor_conv2_grad_output,'seq_conv2_grad_output', t, epochSaveDir)
		    	saveImageDepth(predictor_conv2_grad_output,'dep_conv2_grad_output', t, epochSaveDir)
		    	saveImageSequence(predictor_conv3_grad_output,'seq_conv3_grad_output', t, epochSaveDir)
		    	saveImageDepth(predictor_conv3_grad_output,'dep_conv3_grad_output', t, epochSaveDir)
				]]--
		    	local allin = unpackBuffer(input_encoder_predictor)

		    	allin = makeContiguous(allin:select(2, 1))
		    	saveImageAll(allin, 'allin', t, epochSaveDir)
		    	-- allin = allin:resize(allin:size(1) * allin:size(2), allin:size(2)*allin:size(4))
		    	-- saveImageSequence(input_encoder_predictor, 'dep_input_encoder_predictor', t)
				-- print('sizei of all in ', allin:size())
		    	-- saveImage(allin, 'allin', t)

		    	saveImageAll(unpackBuffer(predictor_conv3.output):select(2,1), 'all_output_conv3', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv2.output):select(2,1), 'all_output_conv2', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(encoder_lstm0.output):select(2,1), 'all_output_lstm0', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(encoder_lstm1.output):select(2,1), 'all_output_lstm1', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv3.lastGradPrevOutput):select(2,1), 'all_lstm1_grad_output', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv2.lastGradPrevOutput):select(2,1), 'all_lstm0_grad_output', t, epochSaveDir)

		    	saveImageAll(unpackBuffer(prediction):select(2,1)[{{2, opt.outputSize},{},{},{}}], 'all_prediction2', t, epochSaveDir)

		    	saveImageAll(unpackBuffer(prediction):select(2,1), 'all_prediction', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv2_grad_output):select(2,1), 'all_conv2_grad_output', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv3_grad_output):select(2,1), 'all_conv3_grad_output', t, epochSaveDir)

		    	saveImageAll(unpackBuffer(target_predictor):select(2,1), 'all_target', t, epochSaveDir)
--[[
		    	saveImageSequence(predictor_conv3.output,'seq_output_conv3', t, epochSaveDir)
		    	saveImageSequence(predictor_conv2.output,'seq_output_conv2', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm0.output,'seq_output_lstm0', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm1.output,'seq_output_lstm1', t, epochSaveDir)

		    	saveImageDepth(predictor_conv3.output,'dep_output_conv3', t, epochSaveDir)
		    	saveImageDepth(predictor_conv2.output,'dep_output_conv2', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm0.output,'dep_output_lstm0', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm1.output,'dep_output_lstm1', t, epochSaveDir)
]]--
		    	saveImageAll(unpackBuffer(predictor_conv3.cells):select(2,1), 'all_cells_conv3', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(predictor_conv2.cells):select(2,1), 'all_cells_conv2', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(encoder_lstm0.cells):select(2,1), 'all_cells_lstm0', t, epochSaveDir)
		    	saveImageAll(unpackBuffer(encoder_lstm1.cells):select(2,1), 'all_cells_lstm1', t, epochSaveDir)
--[[
		    	saveImageSequence(predictor_conv3.cells,'seq_cells_conv3', t, epochSaveDir)
		    	saveImageSequence(predictor_conv2.cells,'seq_cells_conv2', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm0.cells,'seq_cells_lstm0', t, epochSaveDir)
		    	saveImageSequence(encoder_lstm1.cells,'seq_cells_lstm1', t, epochSaveDir)

		    	saveImageDepth(predictor_conv3.cells,'dep_cells_conv3', t, epochSaveDir)
		    	saveImageDepth(predictor_conv2.cells,'dep_cells_conv2', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm0.cells,'dep_cells_lstm0', t, epochSaveDir)
		    	saveImageDepth(encoder_lstm1.cells,'dep_cells_lstm1', t, epochSaveDir)
	]]--	    
			    saveImage(unpackBuffer(prediction), 'output', t, epochSaveDir)

		    	saveImage(unpackBuffer(prediction:add(-prediction:min()):div(prediction:max() - prediction:min())), 'output', t, epochSaveDir)

	--	    	saveImage(unpackBuffer(prediction), 'output', t, epochSaveDir)
		    	saveImage(unpackBuffer(target_predictor), 'target', t, epochSaveDir)

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
	
	local inputSize = 3
	local outputSize = 4
	local bufferStep = 5
	local kernelSizeIn = 3
	local kernelSizeMem = 3
	local stride = 1
	local batchSize = 8 
	local height = 10
	local width = 10
	local defaultType = 'nn.DoubleTensor'

	local net = nn.StepConvLSTM(inputSize, outputSize, bufferStep, kernelSizeIn, kernelSizeMem, stride, batchSize, height, width, defaultType)

	local input = torch.randn(bufferStep * batchSize, inputSize, height, width)
	for i = 1, 3 do
		output = net:forward(input)
		output = net:unpackBuffer(output)
		-- mytester:assertTensorEq(output, net.debugBuffer)
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

function runtest.HadamardMul()
	local inputSize = 3
	local batchSize = 1
	local h = 2
	local w = 2
	local net = nn.HadamardMul(inputSize)
	local input = torch.randn(batchSize * inputSize * h * w):resize(batchSize, inputSize, h, w)
	local output = net:forward(input)
	mytester:assertTensorNe(output, input)
	local original_w = net.weight:clone()
	print(original_w:mean())

	-- cal gradient
	net:zeroGradParameters()
	local p, g = net:getParameters()
	mytester:assert(g:sum() == 0, 'zeroGradParameters fail')

	net:backward(input, output, 1000)
	print('input', input)
	print('output', output)
	mytester:assertTensorNe(output, input, 'forward get output fail')

	mytester:assert(g:sum() ~= 0, 'backward update gradParameters fail')

	-- update parameters
	net:updateParameters(10)
	local p, g = net:getParameters()

	print(p:mean())
	mytester:assertTensorNe(p, original_w, 'updateParameters fail')

	-- forward test2
	net.weight:fill(1)
	output = net:forward(input)
	mytester:assertTensorEq(output, input, 'forward fail')
	net:zeroGradParameters()
	net:backward(input, output:zero()) -- input == output, gradent = 0	
	mytester:assert(g:sum() == 0, 'calculate gradParameters incorrectly')
end


mytester = torch.Tester()
mytester:add(runtest)
-- math.randomseed(os.time())
mytester:run()
