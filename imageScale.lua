local _ = require 'moses'
require 'nn'
require 'image'
require 'torch'
-- require 'extracunn'

local imageScale, parent = torch.class('nn.imageScale', 'nn.Module')

function imageScale:__init(oheight, owidth, mode)
	print('warning: imageScale layer can only be the first layer of the network, since it do not perform backward function')
	self.height = oheight
	self.width = owidth
	self.mode = mode or 'bicubic' -- can be biliear  

end

function imageScale:updateOutput(input)

	local output = torch.Tensor()
	if torch.isTensor(input) then
		local input_clone = input:clone()
		input_clone = input_clone:float()
		assert(input:nDimension() < 5)
		if input:nDimension() == 3 or input:nDimension() == 2 then
			output = image.scale(input_clone, self.height, self.width, self.mode)
			output:typeAs(input_clone)
		elseif input_clone:nDimension() == 4 then
			local batchSize = input_clone:size(1)
			local depth = input_clone:size(2)
			output = torch.Tensor(batchSize, depth, self.height, self.width)
			for i = 1, batchSize do
				output[i] = image.scale(input_clone[i], self.height, self.width, self.mode):typeAs(input_clone)
			end
		end

	else 
		output = {}
		for k = 1, table.getn(input) do
			local subinput = input[k]:clone()
			subinput = subinput:float()
			assert(subinput:nDimension() < 5)
			if subinput:nDimension() == 3 or subinput:nDimension() == 2 then
				output[k] = image.scale(subinput, self.height, self.width, self.mode):typeAs(subinput)
			elseif subinput:nDimension() == 4 then
				local batchSize = subinput:size(1)
				local depth = subinput:size(2)
				output[k] = torch.Tensor(batchSize, depth, self.height, self.width)
				for i = 1, batchSize do
					output[k][i] = image.scale(subinput[i], self.height, self.width, self.mode)
				end
			end
		end
	end

	return output
end

