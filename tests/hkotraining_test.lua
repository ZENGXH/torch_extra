dofile '../../hko/opts-hko.lua'
require 'nn'
require 'math'
dofile '../RecursiveSequential.lua'
dofile '../StepConvLSTM.lua'
dofile '../dataProvider.lua'
dofile '../../hko/routine-hko.lua'
prepareModeldir_Imagedir('test_StepConvLSTM')
defaultType = 'nn.DoubleTensor'

data_path = '../../../ConvLSTM/helper/'	
trainDataProvider = getdataSeq_hko('train', data_path)
validDataProvider = getdataSeq_hko('valid', data_path)
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
    parameters_encoder = parameters_encoder:normal(2, 0.1)
    -- predictor = nn.SelfFeedSequencer( predictor_1 )
    predictor = nn.RecursiveSequential(opt.output_nSeq)
    						:add(predictor_conv2)
    						:add(predictor_conv3)
    						:add(encoder_predictor)
    local parameters_predictor, gradParameters_predictor = predictor:getParameters()
    parameters_predictor = parameters_predictor:normal(2, 0.1)

    print('number of parameters of repeatModel', parameters_encoder:size(1) + parameters_predictor:size(1))

    print('encoder')
  	print(encoder)
  	print('predictor')
  	print(predictor)

-----------------------------------  building model done ----------------------------------


function train()
	epoch = epoch or 1  
	local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
 	if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
	encoder:remember('both')
	encoder:training()
	predictor:remember('both') 
	predictor:training()

	local accErr = 0

	for t =1, 2 do
	    print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	    local tic = torch.tic()

	    local iter = t
	    local trainData = trainDataProvider[t]

	    encoder:zeroGradParameters()
	    encoder:forget()
	    -- encoder:type(defaultType)
	    predictor:zeroGradParameters()
	    predictor:forget()
	    -- predictor:type(defaultType)

	    local input_encoder = torch.Tensor((opt.input_nSeq - 1)*opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    local output_encoder = encoder:forward(input_encoder)


	    print('output_encoder:', output_encoder:size())
	    predictor_conv2.initCell = encoder_lstm0.lastCell
	    predictor_conv3.initCell = encoder_lstm1.lastCell
	    predictor_conv2.initOutput = encoder_lstm0.lastOutput
	    predictor_conv3.initOutput = encoder_lstm1.lastOutput

	    local input_predictor = torch.Tensor(opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)
	    local target_predictor = torch.Tensor((opt.output_nSeq), opt.batchSize, opt.nFiltersMemory[1], opt.inputSizeH, opt.inputSizeW):normal(1, 0.1)

	    predictor:autoForwardAndBackward(input_predictor, target_predictor)

	    -- target = torch.Tensor(opt.output_nSeq, )
	end
end

-- for iter = 1 do
	train()
