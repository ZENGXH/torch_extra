dofile '../../hko/opts-hko.lua'
require 'nn'
require 'math'
dofile '../RecursiveSequential.lua'
dofile '../StepConvLSTM.lua'
dofile '../dataProvider.lua'
dofile '../../hko/routine-hko.lua'
dofile '../utils.lua'
-- --prepareModeldir_Imagedir('test_StepConvLSTM')
defaultType = 'nn.DoubleTensor'
runtest = {}

function runtest.hkotraning()
	if not opt.onMac then
		data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
	else
		data_path = '../../../ConvLSTM/helper/'	
	end
	nn.StepConvLSTM.usenngraph = true
	trainDataProvider = getdataTensor_hko('train', data_path)
	-- validDataProvider = getdataSeq_hko('valid', data_path)
		encoder_lstm0 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq - 1, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph

		encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		opt.input_nSeq - 1, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, 
		                        opt.batchSize, opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph

		encoder_predictor = nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 
	    										opt.decoderKernelSize, opt.decoderKernelSize, 1, 1, 
	    										math.floor(opt.decoderKernelSize / 2), 
	    										math.floor(opt.decoderKernelSize / 2))

	    predictor_conv2 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
		                 		1, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
		                        opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph

	    predictor_conv3 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
		                 		1, opt.kernelSize,
		                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
		                        opt.inputSizeW, opt.inputSizeH, defaultType) -- without nngraph

	    seq = nn.Sequential():add(encoder_lstm0):add(encoder_lstm1)

	    assert(seq:__len() == 2)
	    encoder = seq -- nn.Sequencer(encoder_lstm0)

	    -- print(encoder.modules[1].modules[1])
	    local parameters_encoder, gradParameters_encoder = encoder:getParameters()
	    assert(parameters_encoder, 'parameters_encoder empty') 
	    parameters_encoder = parameters_encoder:normal(0.01, 0.01)
	    -- predictor = nn.SelfFeedSequencer( predictor_1 )
	    predictor = nn.RecursiveSequential(opt.output_nSeq)
	    						:add(predictor_conv2)
	    						:add(predictor_conv3)
	    						:add(encoder_predictor)
	    local parameters_predictor, gradParameters_predictor = predictor:getParameters()
	    parameters_predictor = parameters_predictor:normal(0.01, 0.01)


	    print('number of parameters of repeatModel', parameters_encoder:size(1) + parameters_predictor:size(1))

	    print('encoder')
	  	print(encoder)
	  	print('predictor')
	  	print(predictor)
	  	local p_encoder_lstm0, g_encoder_lstm0 = encoder_lstm0:getParameters()
	  	local p_encoder_lstm1, g_encoder_lstm1 = encoder_lstm1:getParameters()
	  	assert(p_encoder_lstm1:size(1) + p_encoder_lstm0:size(1) == parameters_encoder:size(1) )
	  	local p_predictor_conv3, g_predictor_conv3 = predictor_conv3:getParameters()
	  	mytester:assertTensorEq(p_encoder_lstm0, parameters_encoder:narrow(1, 1,  p_encoder_lstm0:size(1)))
	-----------------------------------  building model done ----------------------------------


	
		epoch = epoch or 1  
		-- local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
	 	-- if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
		encoder:remember('both')
		encoder:training()
		predictor:remember('both') 
		predictor:training()

		local accErr = 0

	    input_encoder = torch.Tensor((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    input_predictor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    target_predictor = torch.Tensor((opt.output_nSeq), opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    if not opt.onMac then
	    	encoder = encoder:cuda()
	    	predictor = predictor:cuda()
	    end

		for t =1, 100 do
		    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
		    local tic = torch.tic()
		    
		    local iter = t
		    local trainData = trainDataProvider[t]
		    

		    input_encoder = trainData[1]:narrow(1, 1, opt.input_nSeq - 1) -- dimension1, from index 1, size = input_nSeq - 1
		    input_predictor = trainData[1]:narrow(1, opt.input_nSeq, 1) 
		    input_encoder:resize((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW)
			input_predictor:resize((1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW)
		    target_predictor = trainData[2]

	    if not opt.onMac then
	    	input_encoder = input_encoder:cuda()
			input_predictor = input_predictor:cuda()
	    	target_predictor = target_predictor:cuda()
	    end
		
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
		    mytester:assert(g_encoder_lstm0:mean() == 0, 'mean is '..g_encoder_lstm0:mean())
		    mytester:assert(g_encoder_lstm1:mean() == 0, 'mean is '..g_encoder_lstm1:mean())
		    
		    -- encoder:type(defaultType)
		    predictor:zeroGradParameters()
		    mytester:assert(g_predictor_conv3:mean() == 0, 'mean is '..g_predictor_conv3:mean())
		    -- mytester:assert(g_encoder_lstm1:mean() == 0, 'mean is '..g_encoder_lstm1:mean())
		    
		    -- predictor:type(defaultType)


		    local output_encoder = encoder:forward(input_encoder)

		    predictor_conv2.initCell = encoder_lstm0.lastCell
		    predictor_conv3.initCell = encoder_lstm1.lastCell
		    predictor_conv2.initOutput = encoder_lstm0.lastOutput
		    predictor_conv3.initOutput = encoder_lstm1.lastOutput


		    output, accErr, gradInput = predictor:autoForwardAndBackward(input_predictor, target_predictor)


		    encoder_lstm0.gradPrevCell = predictor_conv2.gradPrevCell
		    encoder_lstm0:maxBackWard(input_encoder, predictor_conv2.lastGradPrevOutput)

		    encoder_lstm1.gradPrevCell = predictor_conv3.gradPrevCell
		    encoder_lstm1:maxBackWard(encoder_lstm0.output, predictor_conv3.lastGradPrevOutput)

		    predictor:updateParameters(0.00001)
		    encoder:updateParameters(0.00001)

			print("\titer",t, "err:", accErr)
			-- print(output:size())
		    
		    if math.fmod(t, opt.saveInterval) == 1 or t == 1 then
		    	-- print(output:size())
		    	saveImage(output_encoder, 'output_encoder', t)
		    	saveImage(output, 'output', t)
		    	saveImage(target_predictor, 'target', t)
		    	saveImage(gradInput, 'gradInput', t)
		    	-- { Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), }
			end
			local toc = torch.toc(tic)
			print('time used: ',toc)
		end
	

end

mytester = torch.Tester()
mytester:add(runtest)
-- math.randomseed(os.time())
mytester:run()
