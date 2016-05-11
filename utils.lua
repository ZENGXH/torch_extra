-- utils.lua
print(os.date("today is %c"))
curtime = os.date("%d%M")
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

function saveImage(figure_in, name, iter, epochSaveDir, type, numsOfOut)

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
	    	img = img:mul(1/img:max())
--   	if name == 'flow' then
--    		  print('1: mean flow: %.4f flow range: u = %.3f .. %.3f', img[1]:mean(), img[1]:min(), img[1]:max())
--    		  print('2: mean flow: %.4f flow range: u = %.3f .. %.3f', img[2]:mean(), img[2]:min(), img[2]:max())
--    		  print('3: mean flow: %.4f flow range: u = %.3f .. %.3f', img[3]:mean(), img[3]:min(), img[3]:max())
--		end
    	-- img = img:mul(1/img:max()):squeeze()
		    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
		elseif dim == 4 then -- batch, input/outputSize, h, w
			-- print('save 4')
			saveImage(figure_in[1], name, iter, epochSaveDir, type, numsOfOut)
		elseif dim == 5 then -- bufferStep, batch, input/outputSize, h, w
			-- print('save 5', figure_in:size(1) )
			local size = figure_in:size(1) 
			for t = 1, size do
				numsOfOut = t
				saveImage(figure_in[numsOfOut], name, iter, epochSaveDir, type, numsOfOut)
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