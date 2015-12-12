local BinarySVMclassifier, parent = torch.class('nn.BinarySVMclassifier', 'nn.criterion')

function BinarySVMclassifier:__init()
	parent.__init(self)
	self.parameters = 

end

function BinarySVMclassifier:updateOutput(input, target)
	-- target[N, 1]: torch.Tensor(batchSize, 1) = [-1, 1, -1, 1, -1]
	-- input[N, D]: output = input[N, D] x (W[1, D])^T
	-- outputs[N, 1] => [1, -1, 1 ..]
	self.outputs = targets.new():resizeAs(targets) -- [N, 1]
	for batch_i = 1, input:size(1) do
		local input_i = input[batch_i]
		local score = torch.mm(input, self.parameters:t())
		if score >= 0 then 
			self.outputs[batch_i] = 1
		else
			self.outputs[batch_i] = -1
		end
	end
	print('update output to be')
	return self.outputs
end

function BinarySVMclassifier:accGradParameters(input, outputs, w, target)
	-- w: [1, D]
	-- input[N, D]: output = input[N, D] x (W[1, D])^T
	-- penalizedSVMcost = 1/N sum_i max(0, 1 - y_i * input_i * w^T) + lambda * w^2
	-- = 1/N * [max(0, 1 - y_1 * input_1 * W^T) + max(0, 1 - y_1 * input_i * W^T) + ...] 
	-- 	 + lambda * W^T
	-- GradParameters = lambda * w + 1/N [(0 or - y_1 * input_1) + (0 or - y_2 * input_2) + ...]
	local thre = 1
	local lambda = 0.01
	self.gradParameters = w.new():resizeAs(w):fill(0) -- [1, D]
	local subgradParameters = torch.Tensor(input:size(1)):fill(0)
    for batch_i = 1, input:size(1) do
    	local subgradParameters_i = 0
    	if outputs * targets < thre then
			subgradParameters_i = - outputs[batch_i] * input[batch_i] 
			-- outputs[batch_i] should be number
			-- return [1, D] tensor
		end
		subgradParameters = subgradParameters + subgradParameters_i
	end
	self.gradParameters = lambda * w + subgradParameters/input:size(1) 
	-- shape [1, D]
	self.gradParameters = self.gradParameters
	return self.gradParameters
end


function BinarySVMclassifier:updateGradInput(input, gradOutput, w, target)
    -- GradInput = -W/N if y_i * input_i * W^T > 1, else 0
    self.gradInput =  input.new():resizeAs(input):fill(0) -- [N, D]
    local thre = 1
    local resize_flag = 0
    if target:size > 1 then
    	resize_flag = 1
    	target:resize(target:size(1))
    end
    for batch_i = 1, input:size(1) do
    	local gradInput_i = w.new():resize(w):fill(0)
    	if outputs * targets < thre then
    		gradInput_i = - target[batch_i] * w
    	end
    	self.gradInput[batch_i] = gradInput_i
    end
end

function Module:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput, w, target)
   self:accGradParameters(input, outputs, w, target)
   return self.gradInput
end

function Module:backwardUpdate(input, gradOutput, lr)
   self:updateGradInput(input, gradOutput, w, target)
   self:accUpdateGradParameters(input, outputs, w, target)
   return self.gradInput
end
