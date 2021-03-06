-- Hadamard product layer on 2D image
-- input: (batchSize, inputSize, height, width)

-- output: (batchSize, outputSize, height, width)
-- weight: (inputSize, outputSize)
-- 
local HadamardMul, parent = torch.class('nn.HadamardMul', 'nn.Module')

function HadamardMul:__init(inputSize)
   parent.__init(self)
   self.weight = torch.Tensor(inputSize)
   self.gradWeight = torch.Tensor(inputSize)

   self:reset()
end

function HadamardMul:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(1))
   end

   if nn.oldSeed then
      for i=1, self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end
   self.weight:zero()
   return self
end


function HadamardMul:updateOutput(input)
	assert(input:dim() == 4, 'HadamardMul layer require input in dim 4, get '..input:dim())
	self.output = input:clone()
	-- print('weight::: ', self.weight:mean())
	for i = 1, self.weight:size(1) do
   		self.output[{{}, {i}, {}, {}}]:mul(self.weight[i])
   	end
   return self.output
end

function HadamardMul:updateGradInput(input, gradOutput)
    if self.gradInput then
    	self.gradInput:resizeAs(input)
    	for i = 1, self.weight:size(1) do
	        self.gradInput[{{}, {i}, {}, {}}]:add(gradOutput[{{}, {i}, {}, {}}] * self.weight[i])
	    end
    	return self.gradInput
    end
end

function HadamardMul:accGradParameters(input, gradOutput, scale)
    scale = scale or 0.1
    for i = 1, self.weight:size(1) do
	    self.gradWeight[i] = self.gradWeight[i] + scale * torch.dot(gradOutput[{{}, {i}, {}, {}}], input[{{}, {i}, {}, {}}])
   	end
end

-- we do not need to accumulate parameters when sharing

function HadamardMul:clearState()
   return parent.clearState(self)
end

function HadamardMul:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(1), self.weight:size(1))
end