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
	    	img = img/img:max()
--   	if name == 'flow' then
--    		  print('1: mean flow: %.4f flow range: u = %.3f .. %.3f', img[1]:mean(), img[1]:min(), img[1]:max())
--    		  print('2: mean flow: %.4f flow range: u = %.3f .. %.3f', img[2]:mean(), img[2]:min(), img[2]:max())
--    		  print('3: mean flow: %.4f flow range: u = %.3f .. %.3f', img[3]:mean(), img[3]:min(), img[3]:max())
--		end
    	-- img = img:mul(1/img:max()):squeeze()
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
	x = x:view(bufferStepDim, x:size(1)/bufferStepDim, x:size(2), x:size(3), x:size(4))
	return x
end	

function packBuffer(x, bufferStepDim)
	assert(x:dim() == 5)
	-- local bufferSize = x:size(1)/opt.batchSize
	local bufferStepDim = bufferStepDim or nil -- do not need in deed
	if not x:isContiguous() then
		local temp = x.new()
		temp:resizeAs(x):copy(x)
		x = temp
	end
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

function w_init(m)
	print('weight init: xavier')
    m:reset(w_init_xavier(m.nInputPlane*m.kH*m.kW, m.nOutputPlane*m.kH*m.kW))
    if m.bias then
    	m.bias:zero()
    end
end
