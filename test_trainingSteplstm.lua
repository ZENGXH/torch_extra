-- hko_train.lua
--[[
reimplement the model introduced in the paper
encoder: 
- conv0: inputlength -> inputlength; 
- conv1: inputlength -> inputlength; input: conv0.output

decoder:
- conv2: noinput -> outputlength;
- conv3: outputlength -> outputlength; take input: conv2.output
]]

-- test_sgd_selffeed.lua
-- add sgd, for momenttum support
-- add selefeed Sequencer, since that the nn.Repeater feed the same input everytime
-- warp with optical flow

-- ============== below are copied from test_subsample_sellfeed_sgd.lua =============== 

-- warp with optical flow
print("info: this model is trying to re-experiment")
require 'torch'
dofile('./hko/opts-hko.lua') 
print(opt)
---- always load opts first!

unpack = unpack or table.unpack

onMac = opt.onMac
saveOutput = true
resumeFlag = false
resumeIter = 1
verbose = true
display = false

local saveInterval = 5

require 'paths'
gpuflag = opt.gpuflag
wrapOpticalFlow = false
underLowMemory = false
local std = opt.parametersInitStd 

local c = require 'trepl.colorize'

require 'torch_extra/modelSave'
require 'nn'
require 'math'
require 'image'
require 'optim'
-- require 'ConvLSTM'
-- require 'display_flow'
require 'DenseTransformer2D'
require 'rnn'
-- require 'ConvLSTM_NoInput'
require 'stn'
require 'torch_extra/imageScale'
dofile 'torch_extra/dataProvider.lua'
-- dofile 'flow.lua'

typeTensor = torch.Tensor():float()
defaultType = typeTensor:type()


dofile 'hko/routine-hko.lua' 

if not opt.onMac then
	opt.gpuflag = true
	require 'cunn'
	require 'cutorch'
	typeTensor = typeTensor:cuda()
	defaultType = typeTensor:type()
	cutorch.setHeapTracking(true)
    cutorch.setDevice(1)
    local gpuid = 1
    gpuid = selectFreeGpu()
	data_path = '/csproject/dygroup2/xiaohui/ConvLSTM/helper/'
else
	data_path = '../ConvLSTM/helper/'	
end

function typeChecking(x)
	if not defaultType then 
		return
	end
	
	assert(x,"input empty")
	if(torch.isTensor(x)) then
		-- print(x)
		-- print(x:type())
		assert(x:type() == defaultType, "require ")
	elseif torch.type(x) == 'table' then
		typeChecking(x[1])
	elseif torch.isTypeOf(x, 'nn.Module') then
		assert(x._type == defaultType, "requiring")
		
	end
end	

dofile 'torch_extra/SelfFeedSequencer.lua'
dofile 'torch_extra/StepConvLSTM.lua'


prepareModeldir_Imagedir('test_hko_ori')

-- 
print('type:',typeTensor)

-- load data: 

trainDataProvider = getdataSeq_hko('train', data_path)
validDataProvider = getdataSeq_hko('valid', data_path)

-- dataLoad = {intputTable, outputTable}
-- darasetSeq_valid = getdataSeq_hko('valid', data_path)

print('==> training model')
checkMemory()
torch.manualSeed(opt.seed)  

------- ================
local eta0 = 1e-6
local eta = opt.eta
local errs= 0

iter = 0

scaleDown = nn.imageScale(opt.inputSizeH, opt.inputSizeW, 'bicubic'):type(defaultType)
scaleUp = nn.imageScale(opt.imageH, opt.imageW, 'bicubic'):type(defaultType)-- :typeAs(typeT)


if resumeFlag then
	print('==> load model, resume traning')
  	encoder = torch.load(modelDir..'encoder-iter-'..resumeIter..'.bin')
  	predictor = torch.load(modelDir..'predictor-iter-'..resumeIter..'.bin')
  	decoder = torch.load(modelDir..'decoder-iter-'..resumeIter..'.bin')
  	print('encoder')
  	print(encoder)
  	print('predictor')
  	print(predictor)
  	print('decoder')
  	print(decoder)
else
	print('==> build model for traning')

	print(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, 
	                        opt.batchSize, opt.inputSizeW, opt.inputSizeH)
--------------------------------------  building model -----------------------------
-- ============== above are copied from test_subsample_sellfeed_sgd.lua =============== 

	encoder_lstm0 = nn.StepConvLSTM(opt.nFiltersMemory[1], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, 
	                        opt.batchSize, opt.inputSizeW, opt.inputSizeH) -- without nngraph
local a, b = encoder_lstm0:getParameters()

	encoder_lstm1 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.input_nSeq - 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, opt.inputSizeW, opt.inputSizeH):type(defaultType) -- without nngraph

	encoder_predictor = nn.SpatialConvolution(opt.nFiltersMemory[1], opt.nFiltersMemory[2], 
    										opt.decoderKernelSize, opt.decoderKernelSize, 1, 1, 
    										math.floor(opt.decoderKernelSize / 2), 
    										math.floor(opt.decoderKernelSize / 2)):type(defaultType) 
    predictor_conv2 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.output_nSeq + 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
	                        opt.inputSizeW, opt.inputSizeH) -- without nngraph

    predictor_conv3 = nn.StepConvLSTM(opt.nFiltersMemory[2], opt.nFiltersMemory[2],  -- 5, 15?
	                 		opt.output_nSeq + 1, opt.kernelSize,
	                        opt.kernelSizeMemory, opt.stride, opt.batchSize, 
	                        opt.inputSizeW, opt.inputSizeH) -- without nngraph
    seq = nn.Sequential():add(encoder_lstm0)
    					:add(encoder_lstm1)
    assert(seq:__len() == 2)
    encoder = nn.Sequencer(seq)
    -- print(encoder.modules[1].modules[1])
    local parameters_encoder, gradParameters_encoder = encoder_lstm1:getParameters()
 	-- print(opt.paraInit, setDevice)

    parameters_encoder:normal(opt.paraInit, std)
    
    -- print(	opt.decoderKernelSize / 2)
    decoder = nn.SpatialConvolution(opt.nFiltersMemory[2], opt.nFiltersMemory[1], 
    			opt.decoderKernelSize, opt.decoderKernelSize, 1, 1, 
    			math.floor(opt.decoderKernelSize / 2), 
    			math.floor(opt.decoderKernelSize / 2))

    predictor_1 = nn.Sequential()
    						:add(encoder_predictor)
    						:add(predictor_conv2)
    						:add(predictor_conv3)
    						:add(decoder)
    predictor = nn.SelfFeedSequencer( predictor_1 ):type(defaultType)

    local parameters_predictor, gradParameters_predictor = predictor:getParameters()
    parameters_predictor:normal(opt.paraInit, std)
    print(parameters_predictor:type(), "***********")
    -- local parameters_decoder, gradParameters_decoder = decoder:getParameters()
    -- parameters_decoder, gradParameters_decoder:normal(opt.paraInit, std)

    print('number of parameters of repeatModel', parameters_encoder:size(1) + parameters_predictor:size(1))
	assert(torch.isTypeOf(encoder_lstm0, 'nn.Module'))
    print('encoder')
  	print(encoder)
  	print('predictor')
  	print(predictor)
  	print('decoder')
  	print(decoder)

-----------------------------------  building model done ----------------------------------
end

function train()
	encoder:float()
	print('encoder type',encoder._type)
	print(encoder)
	print("====")
	typeChecking(encoder_lstm0.outputs)
  	typeChecking(encoder)
  	typeChecking(predictor)
  	typeChecking(decoder)

	epoch = epoch or 1  
	local epochSaveDir = imageDir..'train-epoch'..tostring(epoch)..'/'
 	if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
  typeChecking(encoder_lstm0.outputs)
	--encoder_0:remember('eval')
	--encoder_0:evaluate()
	encoder:remember('both')
	typeChecking(encoder_lstm0.outputs)
	encoder:training()
	typeChecking(encoder_lstm0.outputs)
	encoder:forget()
	encoder:float()

	typeChecking(encoder_lstm0.outputs)
	predictor:remember('both') 
	predictor:training()
	predictor:forget()
encoder:float()
	local accErr = 0

	for t =1, opt.trainIter do
	  	typeChecking(encoder)
	  	typeChecking(predictor)
	  	typeChecking(decoder)
	    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	    local tic = torch.tic()

	    local iter = t
	    local trainData = trainDataProvider[t]
		typeChecking(encoder_lstm0.outputs)

	    encoder:zeroGradParameters()
	    encoder:forget()
	    encoder:float()

	     		typeChecking(encoder_lstm0.outputs)

	    predictor:zeroGradParameters()
	    predictor:forget()

	    local inputTable = trainData[1]
	    -- assert(baseLSTMCopy.step == 1)
	    ----------------- hard code input -----------
	    
		-- rescale for training
	    --print('before scaling: ', inputTable)
	    inputTable = scaleDown:forward(inputTable)

	    encoderInput = {}
	    local totalLength = table.getn(inputTable)
	    local cur = 1


    	for i = 1, table.getn(inputTable) - 1 do
    		-- inputTable[i] = inputTable[i]:type(defaultType)
    		encoderInput[i] = inputTable[i]:type(defaultType)
    	end


	    -- print('inputTable', inputTable)
	    -- print('encoder_0 input', encoderInput)

	    predictorInput = {}
	    

    	predictorInput[1] = inputTable[opt.input_nSeq]:type(defaultType)
		inputTable = {}
	    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
		    predictorInput[i + 1] = torch.Tensor():type(defaultType):resizeAs(predictorInput[1]):fill(0)
		    -- predictorInput[i + 1] = torch.Tensor():type(defaultType):resizeAs(predictorInput[1]):fill(0)
		end			

	    --- data preparation done ---

	    --- start forward ---
	    -- print('*****************	')
	    -- print(encoderInput[1]:type())
	    -- assert(cunn)
	    if cunn then
	    	encoder  = encoder:cuda()
	    	p,pp = encoder:getParameters()
	    	print(p:type())
	    end

 	    -- typeChecking(inputTable)
		typeChecking(encoderInput)
		typeChecking(predictorInput)

		typeChecking(encoder_lstm0.outputs)

		encoder_lstm0:forward(encoderInput[1])

	    local output_encoder = encoder:forward(encoderInput)

	    -- copy cell state method 1: allocate new memory
	  
	    predictor_conv2.userPrevOutput = encoder_lstm0.outputs[opt.input_nSeq - 1]:clone() -- :type(defaultType)
	    predictor_conv2.userPrevCell = encoder_lstm0.cells[opt.input_nSeq - 1]:clone() -- :type(defaultType)

	    predictor_conv3.userPrevOutput = encoder_lstm1.outputs[opt.input_nSeq - 1]:clone() -- :type(defaultType)
	    predictor_conv3.userPrevCell = encoder_lstm1.cells[opt.input_nSeq - 1]:clone() -- :type(defaultType)
		typeChecking(predictor_conv2.userPrevOutput)
		typeChecking(predictor_conv2.userPrevCell) 
--[[	   
	    predictor_conv2.outputs[1] = encoder_lstm0.outputs[opt.input_nSeq - 1]
	    predictor_conv2.cells[1] = encoder_lstm0.cells[opt.input_nSeq - 1]
predictor_conv2.step = 2
	    predictor_conv3.outputs[1] = encoder_lstm1.outputs[opt.input_nSeq - 1]
	    predictor_conv3.cells[1] = encoder_lstm1.cells[opt.input_nSeq - 1]
predictor_conv2.step = 2
predictor.OutputStep_start = 2
 ]]
---predictorInput_ = {predictorInput[1], predictor_conv2.cells[1], predictor_conv2.outputs[1]}
-- print(predictorInput,"======= input predictor =======")
--		print(predictor.modules)
		print('predictor_conv2.modules')
--		print(predictor_conv2.modules)
predictor:float()
	    output = predictor:forward(predictorInput)


		local criterion = nn.SequencerCriterion(nn.MSECriterion(1)):type(typeTensor:type())	     
		local target = trainData[2]
	    -- local targetSeq = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.fakeDepth, opt.inputSizeH, opt.inputSizeW):typeAs(typeT)

		--print('criterion input: ')
		--print('output', output)
		--print('target', target)
		--print('inputTable',inputTable)

--lstm = repeatModel.modules[1].modules[1].modules[1].modules[1].modules[2].modules[1].modules[1]
-- flow = repeatModel.modules[1].modules[1].modules[1].modules[1].modules[2].modules[2].modules[1]

		target = scaleDown:forward(target)

    	for i = 1, table.getn(target) do
    		target[i] = target[i]:type(defaultType)
    	end
	    

		local criterion = nn.SequencerCriterion(nn.MSECriterion(1)):type(defaultType)
		-- print(output[1]:type())
		-- print(target[1]:type()	)
		-- assert(output[1]:type() == target[1]:type())
		typeChecking(output)
		typeChecking(target)

		err = criterion:forward(output, target)
		accErr = err + accErr
		print("\titer",t, "err:", err)

		if t < 10 then checkMemory('\ncriterion start bp <<<<') end

		local gradOutput = criterion:backward(output, target)


	    if saveOutput and math.fmod(t , saveInterval) == 1 or t == 1 then
	    	saveImage(output, 'output', iter, epochSaveDir, 'output')
	    	saveImage(target, 'target', iter, epochSaveDir, 'output')
	    	-- { Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), Tensor(batch * depth * h * w), }
		end

		if t < 10 then checkMemory('criterion backward done <<<<') end
		--------------************-------------
		for i = 1, #predictorInput - 1 do
			predictorInput[i] = output[i + 1]
		end	
		--------------************-------------
	    target = {}
		output = {}



		-- predictor = predictor:type(defaultType)	
		typeChecking(predictor)
		typeChecking(gradOutput)
--local p,pp = predictor:getParameters()
--print(p:size())
--print(pp:size())
--		print(predictorInput,'predictorInput')
--		print(gradOutput,'gradOutput')
--		print(predictor)
		predictor:updateGradInput(predictorInput, gradOutput)
		predictor:updateParameters(opt.lr)
		predictor:zeroGradParameters()

		-- TODO : UPDATE PARAMETERS OF ENCODER
	    -- local a, b = flowGridGenerator:getParameters()
		if t < 10 then checkMemory('\nconv_4 bp done ') end



--[[
+function SeqLSTM:clearState()
 +   self.cell:set()
 +   self.gates:set()
 +   self.buffer1:set()
 +   self.buffer2:set()
 +   self.buffer3:set()
 +   self.grad_a_buffer:set()
 +
 +   self.grad_c0:set()
 +   self.grad_h0:set()
 +   self.grad_x:set()
 +   self._grad_x = nil
 +   self.output:set()
 +   self._output = nil
 +   self.gradInput = nil
 +end
]]
	    ---------------
		encoder_lstm0.userNextGradCell = predictor_conv2.userGradPrevCell:clone():type(typeTensor:type())
		encoder_lstm0.gradPrevOutput = predictor_conv2.userGradPrevOutput:clone():type(typeTensor:type())

		predictor_conv2:forget()
		predictor_conv2:clearState()

		encoder_lstm1.userNextGradCell = predictor_conv3.userGradPrevCell:clone():type(typeTensor:type())
		encoder_lstm1.gradPrevOutput = predictor_conv3.userGradPrevOutput:clone():type(typeTensor:type())

		predictor_conv3:forget()
		predictor_conv3:clearState()

	    encoder:updateParameters(opt.lr)
		if math.fmod(t, 80)  == 1 then
			print('accE rror of ', 80 * opt.batchSize, 'is >>>>>> ', accErr)
			accErr = 0
		end
	--	print('encoder_0 bp done:')
		if t < 10 then checkMemory("backward done") end
		local toc = torch.toc(tic)
		print('time used: ',toc)

		if t < 10 then checkMemory() end
--		if  t == opt.trainIter / 2 then
--			print('model saved')
--			saveModel(modelDir, encoder, 'encoder_0_middle', epoch)
--		   	saveModel(modelDir, predictor, 'repeatModel_middle', epoch)
--	    end
	    -- encoder = encoder:type(typeTensor:type())
	    -- predictor = predictor:type(typeTensor:type())
	end

	epoch = epoch + 1
end


function valid()
	local totalerror = 0
	local epochSaveDir = imageDir..'valid/'
	if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
-------------

	encoder:remember('both')
	encoder:evaluate()
	encoder:forget()

	predictor:remember('both') 
	predictor:evaluate()
	predictor:forget()
	
	local accErr = 0

	for t =1, opt.validIter do
	    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	    local tic = torch.tic()

	    local iter = t
	    local validData = validDataProvider[t]

	    encoder:zeroGradParameters()
	    encoder:forget()
	    predictor:zeroGradParameters()
	    predictor:forget()

	    local inputTable = validData[1]
	    -- assert(baseLSTMCopy.step == 1)
	    ----------------- hard code input -----------
	    
		-- rescale for training
	    --print('before scaling: ', inputTable)
	    inputTable = scaleDown:forward(inputTable)

	    encoderInput = {}
	    local totalLength = table.getn(inputTable)
	    local cur = 1

	    if opt.gpuflag then 
	    	for i = 1, table.getn(inputTable) - 1 do
	    		encoderInput[i] = inputTable[i]:cuda()
	    	end
	    	encoder:cuda()
	    else 
	    	for i = 1, table.getn(inputTable) - 1 do
-- ## reverse input order
-- ## rotate input image in dataprovider
	    		encoderInput[i] = inputTable[totalLength - i]
	    		cur = i
	    		print(totalLength - i)
	    	end
	    end
	    -- print('inputTable', inputTable)
	    -- print('encoder_0 input', encoderInput)

	    predictorInput = {}
	    
	    if opt.gpuflag then 
	    	predictorInput[1] = inputTable[opt.input_nSeq]:cuda()
			inputTable = {}
		    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
			    predictorInput[i + 1] = torch.Tensor():cuda():resizeAs(predictorInput[1]):fill(0)
			end			
	    else 
	    	--inputTable4[1] = inputTable[opt.input_nSeq]
	    	predictorInput[1] = inputTable[cur - 1]
	    	inputTable = {}
		    for i = 1, opt.output_nSeq - 1 do -- fill in dummpy input
			    predictorInput[i + 1] = torch.Tensor():resizeAs(predictorInput[1]):fill(0)
			end
	    end
	    --- data preparation done ---

	    --- start forward ---
	    local output_encoder = encoder:forward(encoderInput)

	    -- copy cell state method 1: allocate new memory
	  
	    predictor_conv2.userPrevOutput = encoder_lstm0.outputs[opt.input_nSeq - 1]:clone()
	    predictor_conv2.userPrevCell = encoder_lstm0.cells[opt.input_nSeq - 1]:clone()

	    predictor_conv3.userPrevOutput = encoder_lstm1.outputs[opt.input_nSeq - 1]:clone()
	    predictor_conv3.userPrevCell = encoder_lstm1.cells[opt.input_nSeq - 1]:clone()

	    output = predictor:forward(predictorInput)


		local criterion = nn.SequencerCriterion(nn.MSECriterion(1)):type(typeTensor:type())	     
		local target = validData[2]

		target = scaleDown:forward(target)

	    if opt.gpuflag then 
	    	for i = 1, table.getn(target) do
	    		target[i] = target[i]:type(typeTensor:type())
	    	end
	    end

		local criterion = nn.SequencerCriterion(nn.MSECriterion(1)):type(typeTensor:type())


		err = criterion:forward(output, target)
		accErr = err + accErr
		print("\titer",t, "err:", err)

		if t < 10 then checkMemory('\ncriterion start bp <<<<') end

		local gradOutput = criterion:backward(output, target)

		if t < 10 then checkMemory('criterion backward done <<<<') end
		--------------************-------------
		for i = 1, #predictorInput - 1 do
			predictorInput[i] = output[i + 1]
		end	
		--------------************-------------
	    target = {}
		output = {}

		local toc = torch.toc(tic)
		print('time used: ',toc)

	end
end

-------

for expoch = 1, opt.maxEpoch do
	train()
	print('run validation')
	valid()
end



