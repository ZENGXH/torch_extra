-- imageConvertor.lua
-- TODO:1111111111111111
-- convert table to images
-- convert 4D dimension tensor to images
-- convert weights to image set
-----

function saveImage(figure_in, name, iter, epochSaveDir, type)
	if epochSaveDir == nil then
		epochSaveDir = "./temp_saveImage/"
		if not paths.dirp(epochSaveDir) then os.execute('mkdir -p ' .. epochSaveDir) end
	end

	if(type == "weight") then saveImage_weight(figure_in, name, iter, epochSaveDir)

	if(torch.Tensor(firgure_in)) then
		saveImage_tensor(figure_in, name, iter, epochSaveDir, type)
	else
		saveImage_table(figure_in, name, iter, epochSaveDir, type)
	end

	local numsOfOut = numsOfOut or 0
	local figure = torch.Tensor()
	if torch.isTensor(figure_in)  then 
    	local img = figure_in:clone():typeAs(typeT)
    	img = img:mul(1/img:max()):squeeze()
--   	if name == 'flow' then
--    		  print('1: mean flow: %.4f flow range: u = %.3f .. %.3f', img[1]:mean(), img[1]:min(), img[1]:max())
--    		  print('2: mean flow: %.4f flow range: u = %.3f .. %.3f', img[2]:mean(), img[2]:min(), img[2]:max())
--    		  print('3: mean flow: %.4f flow range: u = %.3f .. %.3f', img[3]:mean(), img[3]:min(), img[3]:max())
--		end
    	-- img = img:mul(1/img:max()):squeeze()
	    image.save(epochSaveDir..'iter-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png',  img)
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

function saveImage_table(figure_table, name, iter, epochSaveDir, type, numsOfOut)
	numsOfOut = numsOfOut or 0
	for numsOfOut = 1, #(figure_table) do
		saveImage_tensor(figure_table[numsOfOut], name, iter, epochSaveDir, type, numsOfOut)
	end
end

function saveImage_tensor(figure_in, name, iter, epochSaveDir, type, numsOfOut)
	numsOfOut = numsOfOut or 0
	if （(figure_in:size() == 2) or (figure_in:size() == 3 and firgure_in:size(1) == 1)） then
		local img = figure_in:clone()
		_save(img, name, iter, epochSaveDir, type, numsOfOut)
		return

	elseif (figure_in:size() == 3 and firgure_in:size(1) == 3) then
		local img = figure_in:clone()
		_save(img, name, iter, epochSaveDir, type, numsOfOut)
		return
	elseif (figure_in:size() == 3) then  -- depth != 3 or 1
		local img = figure_in[1] -- select the first batch
		_save(img, name, iter, epochSaveDir, type, numsOfOut)
		return
	elseif (figure_in:size() == 4) then
		-- first dimension is batch
		-- second dimension is depth
		local img = figure_in[1][1]
		_save(img, name, iter, epochSaveDir, type, numsOfOut)
		return
	else
		print("image save not  implement: ", figure:size())
	end
end

function saveImage_weight(figure_in, name, iter, epochSaveDir) -- normaly, weights are tensor
	local img = figure_in:clone()
	if(figure_in:size() == 3) then
		if(figure_in:size(2) == figure_in:size(3)) then
			local img = figure_in:clone()
			local width = figure_in:size(2)
			local depth = figure_in:size(1)
	        _savegroup(img, name, iter, epochSaveDir, depth)
        elseif(figure_in:size(2) == figure_in:size(1)) then
			local img = figure_in:clone()
			local width = figure_in:size(2)
			local depth = figure_in:size(3)
			img = img:transpose(1, 3)
	        _savegroup(img, name, iter, epochSaveDir, depth)
        end        	
    elseif(figure_in:size() == 4) then
    	local img = figure_in:clone()
		if(figure_in:size(3) == figure_in:size(4)) then
			local img = figure_in:clone()
			local width = figure_in:size(3)
			local depth = figure_in:size(1) * figure_in:size(2)
			img = img:resize(depth, width, width)
	        _savegroup(img, name, iter, epochSaveDir, depth)

		elseif(figure_in:size(1) == figure_in:size(2)) then
			local img = figure_in:clone()
			local width = figure_in:size(2)
			local depth = figure_in:size(3) * figure_in:size(4)
			img = img:resize(width, width, depth)
			img = img:transpose(1, 3)
	        _savegroup(img, name, iter, epochSaveDir, depth)
		elseif(figure_in:size(1) == figure_in:size(3)) then
			img = img:transpose(1, 4)
			saveImage_weight(img, name, iter, epochSaveDir)	  
		elseif(figure_in:size(1) == figure_in:size(4)) then
			img = img:transpose(1, 3)
			saveImage_weight(img, name, iter, epochSaveDir)  
	    end
    end
end


function _savegroup(firgure_in, name, iter, epochSaveDir, depth)
	local img = firgure_in:clone()
	local imgName = epochSaveDir..timeStamp()..'-it-'..tostring(iter)..'-'..name..'.png'
    local img_group = image.toDisplayTensor{input = img,
                           padding = 2,
                           nrow = math.floor(math.sqrt(depth)),
                           symmetric = true}
    image.save(imgName, img_group / img_group:max())
end	

function _save(figure_in, name, iter, epochSaveDir, type, numsOfOut)
	numsOfOut = numsOfOut or 0
	local img = figure_in:clone()
	local imgName = epochSaveDir..timeStamp()..'-it-'..tostring(iter)..'-'..name..'-n'..tostring(numsOfOut)..'.png'
    image.save(imgName, img/img:max())
end




function timeStamp()
  return './data/hko/'..os.date("%m%d%H%M%S")..'-os'
end