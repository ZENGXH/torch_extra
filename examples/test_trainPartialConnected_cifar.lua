require 'xlua'
require 'optim'
require 'nn'
require 'save_model'
--dofile '../PartialConnected.lua'
--require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs/train_pcfast")      subdirectory to save logs
   -b,--batchSize             (default 20)          batch size
   -r,--learningRate          (default 0.0001)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 10)          epoch step
   --model                    (default ../pcfast_cifar)     model name
   --max_epoch                (default 30)           maximum number of iterations
   -r, --resume_flag              (default 1)  if 0, then read model from save path
]]

print(opt)

last_loss = -10 -- init loss to a negetive value

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      -- randomly choose some image to filp
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output = input
    return self.output
  end
end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
-- DataAugmentation:


--model:add(nn.Copy('torch:FloatTensor','torch:CudaTensor'):cuda()) --:cuda() make a variable CudaTensor
-- IMPORTANT: CAST MODEL TO FLOAT!

-- load model from file
if opt.resume_flag == 0 then
  print('resume trainig')
  model = torch.load(opt.save..'/model.net')
else
  print('creating new model')
  model:add(nn.BatchFlip():float())
  model:add(dofile(opt.model..'.lua'))
end

--model:add(dofile('models/'..opt.model..'.lua'):cuda())
model:get(2).updateGradInput = function(input) return end
print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion()
--criterion = nn.CrossEntropyCriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  -- local targets = torch.CudaTensor(opt.batchSize)
  local targets = torch.FloatTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()

  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    --------------
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      model = model:float()
      local outputs = model:forward(torch.FloatTensor(inputs))
      targets = targets:float()
      outputs = outputs:float()
--	print('forward done, output', outputs)
      -- print prediction
      -- maxi,index = torch.max(outputs, 2)
      -- print(index)
      -- print(targets)
      -- if t%10 == 0 then print(outputs) end
      --print(targets)

      criterion = criterion:float()
      local f = criterion:forward(outputs, targets)
	print('loss', f)
      local df_do = criterion:backward(outputs, targets)
--	print('start backward')
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)
      
      if t%30 == 0
         then
            confusion:updateValids() 
            if last_loss > 0 then
                local cur_loss = confusion.totalValid * 100
                if cur_loss - last_loss > 5 then
                   local filename = paths.concat(opt.save, 'model.net')
                   print('increase by 5!') 
                   saveModel(model,filename) -- save model call function in 'save_model.lua'
                end
            end
            last_loss = confusion.totalValid * 100
            print(confusion.totalValid * 100)
            print(confusion)      
         end
      
      return f,gradParameters
    
    end
    -----------------------------

    optim.sgd(feval, parameters, optimState)

  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 125
  for i=1,provider.testData.data:size(1),bs do
    local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
   -- testLogger:add{train_acc, confusion.totalValid * 100, confusion}
    testLogger:add{train_acc, confusion.totalValid*100}
   --testLogger:style{'-','-'}
   --  testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
--  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
 --  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


