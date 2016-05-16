-- utils.lua
print(os.date("today is %c"))
curtime = os.date("%d%H")
print('current time: ', curtime)

function typeChecking(x, defaultType)
	assert(defaultType, 'defaultType must provided if you want typeChecking')
	assert(x, "input of typeChecking empty")
	if(torch.isTensor(x)) then
		-- print(x)
		-- print(x:type())
		assert(x:type() == defaultType, "type typeChecking fail get: "..x:type()..' and default:'..defaultType)
	elseif torch.type(x) == 'table' then
		typeChecking(x[1], defaultType)
	elseif torch.isTypeOf(x, 'nn.Module') then
		assert(x._type == defaultType, "requiring")
		
	end
end	

function saveImageDepth(figure_in, name, iter, epochSaveDir)

	-- local length = figure_in:size(1)/opt.batchSize
	-- ori: bufferSize*batchSize, depth, h, w
	-- unpacl: bufferSize, batchSize, depth, h, w
		-- select the last frame, first batch
		-- depth, h, w
	local figure_in = unpackBuffer(figure_in):select(1, figure_in:size(1)/opt.batchSize):select(1, 1)
	figure_in = makeContiguous(figure_in)
	assert(figure_in:size(2) == figure_in:size(3))

	figure_in = figure_in:view(figure_in:size(1) * figure_in:size(2), figure_in:size(3))
	saveImage(figure_in, name, iter, epochSaveDir)
end

function makeContiguous(figure_in)
	if not figure_in:isContiguous() then
		-- print('not contiguous')
		local temp = figure_in.new()
		temp:resizeAs(figure_in):copy(figure_in)
		figure_in = temp
		return temp
	else
		-- print('contiguous!')
		return figure_in
	end
end


function saveImageSequence(figure_in, name, iter, epochSaveDir)
	assert(figure_in:dim() == 4, 'figure_in dim'..figure_in:dim()..name)

	-- assert(figure_in:size(2) == opt.batchSize)

	-- select the first batch, first depth
	figure_in = unpackBuffer(figure_in):select(2, 1):select(2, 1)
	figure_in = makeContiguous(figure_in)
	assert(figure_in:size(2) == figure_in:size(3))
	
	figure_in = figure_in:view(figure_in:size(1) * figure_in:size(2), figure_in:size(3))
	saveImage(figure_in, name, iter, epochSaveDir)
end

function saveImage(figure_in, name, iter,epochSaveDir, type, numsOfOut, pack)
	local pack = pack or false
	local figure_in = figure_in
	if pack and figure_in:dim() == 4 then
		figure_in = unpackBuffer(figure_in)
	end
	local numsOfOut = numsOfOut or 0
	local figure = torch.Tensor()
	local epochSaveDir = epochSaveDir or './data/'..curtime..'temp/'
	if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
	local type = type or 'output'
	local iter = iter or 0
	local name = name or 'output'

	if torch.isTensor(figure_in)  then 
		local dim = figure_in:dim()
		if dim == 3 then -- input/outputSize, h, w
			-- print('save 3')
	    	local img = figure_in:clone()
	    	if img:size(2) ~= 3 then
		    	img = img:narrow(1, 1, 1)
		    end
		    --if name == 'output'  then
		    --	img = img:mul(255)
	    	-- img:div(img:max())

		    --else
		    img:add(-img:min()):div(img:max() - img:min())
			--end

--   	if name == 'flow' then
--    		  print('1: mean flow: %.4f flow range: u = %.3f .. %.3f', img[1]:mean(), img[1]:min(), img[1]:max())
--    		  print('2: mean flow: %.4f flow range: u = %.3f .. %.3f', img[2]:mean(), img[2]:min(), img[2]:max())
--    		  print('3: mean flow: %.4f flow range: u = %.3f .. %.3f', img[3]:mean(), img[3]:min(), img[3]:max())
--		end
    	-- img = img:mul(1/img:max()):squeeze()

		    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
		elseif dim == 2 then
			-- print('save!!!!')
	    	local img = figure_in:clone()
	    	-- img:div(img:max())
		    img:add(-img:min()):div(img:max() - img:min())
		    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
		elseif dim == 4 then -- batch, input/outputSize, h, w
			-- print('save 4')
			saveImage(figure_in[1], name, iter, epochSaveDir, type, numsOfOut, false)
		elseif dim == 5 then -- bufferStep, batch, input/outputSize, h, w
			-- print('save 5', figure_in:size(1) )
			local size = figure_in:size(1) 
			for t = 1, size do
				numsOfOut = t
				saveImage(figure_in[numsOfOut], name, iter, epochSaveDir, type, numsOfOut, false)
			end
		end

    elseif type == 'output' then------------- table --------------	
    	for numsOfOut = 1, table.getn(figure_in) do
    		------- choose the first batch along 
    		local img = figure_in[numsOfOut][1]:clone()  --- 
    		saveImage(img, name, iter, epochSaveDir, type, numsOfOut)
    	end
    else 
     	for numsOfOut = 1, table.getn(figure_in) do
    		------- choose the first batch along 
    		local img = figure_in[numsOfOut]:clone()  --- 
    		saveImage(img, name, iter, epochSaveDir, type, numsOfOut)
    	end   	
	end
end



function selectFreeGpu()
	local gpuid = 1
	if opt.onMac then
		return 0
	end
	local freeMemory, totalMemory = cutorch.getMemoryUsage(1)
	print('free',freeMemory/(1024^3))
	local freeMemory2, totalMemory = cutorch.getMemoryUsage(2)
	print('free2',freeMemory2/(1024^3))
	if(freeMemory2 > freeMemory) then 
		cutorch.setDevice(2)
		gpuid = 2
	end
	return gpuid
end

function checkMemory(message, gpuid) 
  	if not opt.onMac then
  		local gpuid = gpuid or 1
  		print(message)
		local freeMemory, totalMemory = cutorch.getMemoryUsage(gpuid)
		print('free',freeMemory/(1024^3))
	else 
		print('not checkMemory')
	end
end

function unpackBuffer(x, bufferStepDim)
	assert(x:dim() == 4)
	-- local bufferSize = x:size(1)/opt.batchSize
	local bufferStepDim = bufferStepDim or x:size(1)/opt.batchSize
	x = makeContiguous(x)
	x = x:view(bufferStepDim, x:size(1)/bufferStepDim, x:size(2), x:size(3), x:size(4))
	return x
end	

function packBuffer(x, bufferStepDim)
	assert(x:dim() == 5)
	-- local bufferSize = x:size(1)/opt.batchSize
	local bufferStepDim = bufferStepDim or nil -- do not need in deed
	x = makeContiguous(x)
	x = x:view(x:size(1) * x:size(2), x:size(3), x:size(4), x:size(5))

	return x
end	

-- weight init adapted from ConvLSTM/weight-init.lua --
-- "Efficient backprop"
-- Yann Lecun, 1998
function w_init_heuristic(fan_in, fan_out)
   return math.sqrt(1/(3*fan_in))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier(fan_in, fan_out)
   return math.sqrt(2/(fan_in + fan_out))
end


-- "Understanding the difficulty of training deep feedforward neural networks"
-- Xavier Glorot, 2010
function w_init_xavier_caffe(fan_in, fan_out)
   return math.sqrt(1/fan_in)
end


-- "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
-- Kaiming He, 2015
function w_init_kaiming(fan_in, fan_out)
   return math.sqrt(4/(fan_in + fan_out))
end



function w_init(m, div, stdset)
	local div = div or 1
	local parameters, g = m:getParameters()
	local assertp = parameters:clone()
	local std = w_init_xavier(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW/div)
	--local std = torch.Tensor(1):fill(std):normal(0, 0.01)[1]
	print('weight init: xavier std: ',std, m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW/div)

	m:reset(std)

    assert(m.weight)
    -- m.weight = m.weight:uniform(-0.008, 0.008)
    if m.bias then
    	m.bias:zero()
    end
    assert(parameters:add(-assertp):mean() ~= 0)
end


function momentumUpdateParameters(net, lr, momentum)
	-- # Momentum update
	-- v = mu * v - learning_rate * dx # integrate velocity
	-- x += v # integrate position
	local parameters, gradParameters = net:getParameters()
	local lr = lr or 1e-4
	local momentum = momentum or 0.9
	if opt.onMac then
		net.v = net.v or torch.Tensor():resizeAs(gradParameters):zero()
	else
		net.v = net.v or torch.Tensor():cuda():resizeAs(gradParameters):zero()
	end
	net.v:mul(momentum)
	net.v:add(-lr, gradParameters)

	parameters:add(net.v)
end

function nestMomentumUpdateParameters(net)
end

function weightVis(net, name, t)
	local parameters, gradParameters = net:getParameters()
	local weight
	if net.weight then
		weight = net.weight:view(net.nInputPlane*net.nOutputPlane, net.kH, net.kW)
	else
		weight = net.module.weight:view(net.nInputPlane*net.nOutputPlane, net.kH, net.kW)
	end

    local dd = image.toDisplayTensor{input=weight,
                           padding=2,
                           nrow=math.floor(math.sqrt(net.nInputPlane*net.nOutputPlane)),
                           symmetric=true}
                           print('weight mean of '..name, weight:mean())
    -- print('dsize', dd:size())
    saveImage(dd, name, t)
end

function saveImageAll(x, name, t, epochSaveDir)
	assert(x:dim() == 4)
	x = makeContiguous(x)
	local row = x:size(1)
	x = x:view(x:size(1), x:size(3)*x:size(2), x:size(4))
    local dd = image.toDisplayTensor{input=x,
                           padding=0,
                           min=0,
                           max=1,
                           nrow= x:size(1),
                           symmetric=false}
    -- print('weight mean of '..name, weight:mean())
    -- print('dsize', dd:size())
    saveImage(dd, name, t, epochSaveDir)
end

function reshape_patch(img, patchSize)
	local patchSize = patchSize or 2
	assert(img:dim() == 4, 'reshape_patch require image in size: batchSize, depth, h, w')
	-- i2 = img
	img = img:reshape(img:size(1), -- B
					img:size(2),   -- D
					img:size(3)/patchSize, -- H/P
					patchSize, 		-- P
					img:size(4)/patchSize, -- W/P
					patchSize)
			:permute(1, 2, 4, 6, 3, 5)
			:reshape(img:size(1), img:size(2) * patchSize * patchSize, img:size(3)/patchSize, img:size(4)/patchSize)
	--[[ network version:
	img = nn.View(batchSize, inputSize, 512/patchSize, patchSize, 512/patchSize, patchSize):forward(img)
	img = nn.Transpose({5,6}, {4,3}, {4,5}):forward(img) -- 2, 4, 256, 256
	img = nn.View(batchSize, inputSize * patchSize * patchSize, 512/patchSize, 512/patchSize):forward(img)
	]]--
	return img
end

function reshape_patch_back(img, patchSize)
	assert(img:dim() == 4, 'reshape_patch require image in size: batchSize, D*P*P, h/P, w/P')

	-- input in size: batchSize(B), inputSize(D) * patchSize(P) * patchSize, height/patchSize, width/patchSize
   img = img:reshape(img:size(1), -- B
					(img:size(2)/patchSize)/patchSize,  -- D
					patchSize, 	  -- P
					patchSize, 	  -- P
					img:size(3),  -- H/P
					img:size(4)	  -- W/P
					) -- now B, D, P, P, H/P, W/P 
   			:permute(1, 2, 5, 3, 6, 4) -- B, D, H/P, P, W/P, P
   			:reshape(img:size(1), -- B
					(img:size(2)/patchSize)/patchSize, -- D
					img:size(3) * patchSize,  -- H
					img:size(4) * patchSize	-- W
					)
 	return img
 end


