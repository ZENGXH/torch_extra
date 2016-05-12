require 'image'

local data_verbose = false

if opt.appliedMask and dataMask == nil then
   require '../torch_extra/dataMask'
   dataMask = dataMask(opt.imageH, opt.imageW, opt.maskStride, opt.maskStride * opt.maskStride) 
end

-- return table{1:inputTable, 2:outputTable}
-- inputTable = {1: inputBatch1, 2: inputBatch2 ...}
function getdataSeq_hko(mode, data_path)
   local mode = mode or "train"
   local data_path = data_path or '../helper/'
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   -- local data_path = 
   -- data size (totalInstances or nsamples=2000?, sequence_length=20, 1, 64, 64)
   local datasetSeq ={}
   gpuflag = opt.gpuflag or false


   -- data = data:float()/255.0 -- to range(0, 1)

   --------------- configuration: -----------------
   -- local std = std or 0.2
   local nsamples = 0
   if mode == "train" then
      nsamples = opt.trainSamples-- data:size(1)
      f = io.open(data_path..'trainseq.txt', 'r')
   elseif mode == "valid" then
      nsamples = opt.validSamples-- data:size(1)
      f = io.open(data_path..'validseq.txt', 'r')
   elseif mode == "test" then
      nsamples = opt.trainSamples-- data:size(1)
      f = io.open(data_path..'testseq.txt', 'r')
   end      
   
   local nseq  = opt.nSeq or nseq-- data:size(2)
   local nrows = opt.inputSizeH or nrows
   local ncols = opt.inputSizeW or nrows
   local nbatch = opt.batchSize or 1

   print (mode .. ' dataload: ' .. nsamples .. ' ' .. nseq .. ' ' .. nrows .. ' ' .. ncols )

   ------------- read the powerful txt file! ------
   local fileList = {}
   
   local id = 1
   for line in f:lines() do
      fileList[id] = line
      id = id + 1
      if id < 100 then
         -- print(line)
      end
   end
   assert(table.getn(fileList) == nseq * nsamples)

   local ids = torch.range(1, nseq * nsamples)
   local seqHeads = torch.Tensor(nsamples)


   if mode == 'train--' then
      print('training mode, shffule heads')
      shuffleID = torch.randperm(nsamples) 
      -- range from 1 to numOfBatches, for shuffle purpose
   else
      shuffleID = torch.range(1, nsamples)
      -- then shuffleBatchID[i] = i
   end   

   for i = 1, nsamples do
      seqHeads[i] = (shuffleID[i] - 1) * nseq + 1 
      --if i < 100 then
      --   print('head :', seqHeads[i])
      --   print(fileList[seqHeads[i]])
      --end
      -- [1, 41, 121, 61, ..] index line number of first frame of the sequence of each samples
   end

   local numsOfBatches = opt.numsOfBatches or torch.floor(nsamples / opt.batchSize) 

   function datasetSeq:size()
      return nsamples
   end

   function datasetSeq:selectSeq(index)
      if simdata_verbose then
         print('selectSeq')
      end
      local input_batch = torch.Tensor(opt.batchSize, opt.input_nSeq, opt.imageDepth, opt.imageH, opt.imageW)
      local output_batch = torch.Tensor(opt.batchSize, opt.output_nSeq, opt.imageDepth, opt.imageH, opt.imageW)

      local inputTable = {}
      local targetTable = {}

      -- index range from 1 to numsOfBatch
      -- for each selection index, jump #opt.batchSize of heads 
      -- if opt.batchSize is 8, then it can be 1, 9, 17, ...
      -- corresponding to 
      -- seqHeads[1], seqHeads[9], seqHeads[17]    
      local enterSeqHeads = (index - 1) * opt.batchSize + 1 


      -- selectIndStart start from 1, nseq*opt.batchSize ... opt.batciSize*nseq samples... 2 * nseq*opt.batchSize
      -- for different selection, jump #opt.batchSize of head, ie jump #opt.batchSize samples
      -- #opt.batchSize indicate how many samples in a batch
      local selectStep = opt.selectStep or 1

      -- start from the entering, increment 1 for each miniBatch when filling one batch 
      local selectIndStart = enterSeqHeads 

      for batch_ind = 1, opt.batchSize do -- filling one batch one by one
         -- start form the seqHeads, increment #selectStep line for each frame in when filling one sequence
         -- line index is ust for reading the #line of the file List, 

         lineInd = seqHeads[selectIndStart]

         ------------- input seq ------------
         for k = 1,  opt.input_nSeq do
            local out = image.load(data_path..'data/'..fileList[lineInd])
            -- print('select start from ',  fileList[lineInd])
            --local out = dataMask:quick4Mask3Dforward(ori)
            -- out = image.rotate(out, 1.5)
            input_batch[batch_ind][k] = out:clone()
            lineInd = lineInd + selectStep
                        --epochSaveDir = 'image/testout/'..'train-epoch'..tostring(0)..'/'
                        --image.save(epochSaveDir..fileList[lineInd],  out)

         end

         ------------- output seq ------------
         for k = 1,  opt.output_nSeq do
            local out = image.load(data_path..'data/'..fileList[lineInd])
            --local out = dataMask:quick4Mask3Dforward(ori)
            -- out = image.rotate(out, 1.5)
            output_batch[batch_ind][k] = out:clone()

            lineInd = lineInd + selectStep
         end
         -- input_batch = input_batch:gt(0.1)
         -- output_batch = output_batch:gt(0.1)
         -- # TODO: make sure selectInd do not go out of bound[now mannuly?]
         selectIndStart = selectIndStart + 1 -- goto next minibatch
      end

      ------------- make it a table -----------
      for k = 1, opt.input_nSeq do
         table.insert(inputTable, input_batch[{{}, {k}, {}, {}, {}}]:select(2,1):reshape(opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW))
      end

      for k = 1, opt.output_nSeq do
         table.insert(targetTable, output_batch[{{}, {k}, {}, {}, {}}]:select(2,1):reshape(opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW))
      end     

         --for numsOfOut = 1, 3 do
         --   print('----------------')
         --   epochSaveDir = 'image/testout/'..'train-epoch'..tostring(0)..'/'
         --   print(inputTable)
         --   name = epochSaveDir..'00iter-'..tostring(index)..tostring(numsOfOut)..'.png'
         --   print(name)
         --   --saveImage(inputTable[n][1], '-aa-input'..n, index, 0, 'output')
         --   image.save(name,  inputTable[numsOfOut][1])

         --end      

      return inputTable, targetTable
   end

   -- dsample(20, 1, 64, 64)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                           local inputTable, targetTable = self:selectSeq(index)
                                           
                                           return {inputTable, targetTable}
                                       end})
   return datasetSeq

end

-- return table{1: inputTensor(), 2: outputTensor()}
-- inputTensor(bufferSize, batchSize, inputSize, height, width)
function getdataTensor_hko(mode, data_path)
   local mode = mode or "train"
   local data_path = data_path or '../helper/'
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   -- local data_path = 
   -- data size (totalInstances or nsamples=2000?, sequence_length=20, 1, 64, 64)
   local datasetSeq ={}
   gpuflag = opt.gpuflag or false


   -- data = data:float()/255.0 -- to range(0, 1)

   --------------- configuration: -----------------
   -- local std = std or 0.2
   local nsamples = 0
   if mode == "train" then
      nsamples = opt.trainSamples-- data:size(1)
      f = io.open(data_path..'trainseq.txt', 'r')
   elseif mode == "valid" then
      nsamples = opt.validSamples-- data:size(1)
      f = io.open(data_path..'validseq.txt', 'r')
   elseif mode == "test" then
      nsamples = opt.trainSamples-- data:size(1)
      f = io.open(data_path..'testseq.txt', 'r')
   end      
   
   local nseq  = opt.nSeq or nseq-- data:size(2)
   local nrows = opt.inputSizeH or nrows
   local ncols = opt.inputSizeW or nrows
   local nbatch = opt.batchSize or 1

   print (mode .. ' dataload: ' .. nsamples .. ' ' .. nseq .. ' ' .. nrows .. ' ' .. ncols )

   ------------- read the powerful txt file! ------
   local fileList = {}
   
   local id = 1
   for line in f:lines() do
      fileList[id] = line
      id = id + 1
      if id < 100 then
         -- print(line)
      end
   end
   assert(table.getn(fileList) == nseq * nsamples)

   local ids = torch.range(1, nseq * nsamples)
   local seqHeads = torch.Tensor(nsamples)


   if mode == 'train--' then
      print('training mode, shffule heads')
      shuffleID = torch.randperm(nsamples) 
      -- range from 1 to numOfBatches, for shuffle purpose
   else
      shuffleID = torch.range(1, nsamples)
      -- then shuffleBatchID[i] = i
   end   

   for i = 1, nsamples do
      seqHeads[i] = (shuffleID[i] - 1) * nseq + 1 
      --if i < 100 then
      --   print('head :', seqHeads[i])
      --   print(fileList[seqHeads[i]])
      --end
      -- [1, 41, 121, 61, ..] index line number of first frame of the sequence of each samples
   end

   local numsOfBatches = opt.numsOfBatches or torch.floor(nsamples / opt.batchSize) 

   function datasetSeq:size()
      return nsamples
   end

   function datasetSeq:selectTensor(index)
      if verbose then
         print('selectSeq')
      end
      local input_batch = torch.Tensor(opt.input_nSeq, opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW)
      local output_batch = torch.Tensor(opt.output_nSeq, opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW)

      index = math.fmod(index, nsamples/opt.batchSize-opt.input_nSeq + 1)


      -- index range from 1 to numsOfBatch
      -- for each selection index, jump #opt.batchSize of heads 
      -- if opt.batchSize is 8, then enterSeqHeads can be 1, 9, 17, ...
      -- corresponding to 
      -- seqHeads[1], seqHeads[9], seqHeads[17]    
      local enterSeqHeads = (index - 1) * opt.batchSize + 1 


      -- selectIndStart start from 1, nseq*opt.batchSize ... opt.batciSize*nseq samples... 2 * nseq*opt.batchSize
      -- for different selection, jump #opt.batchSize of head, ie jump #opt.batchSize samples
      -- #opt.batchSize indicate how many samples in a batch
      local selectStep = opt.selectStep or 1

      -- start from the entering, increment 1 for each miniBatch when filling one batch 
      local selectIndStart = enterSeqHeads 

      for batch_ind = 1, opt.batchSize do -- filling one batch one by one
         -- start form the seqHeads, increment #selectStep line for each frame in when filling one sequence
         -- line index is ust for reading the #line of the file List, 

         lineInd = seqHeads[selectIndStart]

         ------------- input seq ------------
         for k = 1,  opt.input_nSeq do
            local out = image.load(data_path..'data/'..fileList[lineInd])
            -- print('select start from ',  fileList[lineInd])
            --local out = dataMask:quick4Mask3Dforward(ori)
            -- out = image.rotate(out, 1.5)
            input_batch[k][batch_ind] = out:clone()
            lineInd = lineInd + selectStep
                        --epochSaveDir = 'image/testout/'..'train-epoch'..tostring(0)..'/'
                        --image.save(epochSaveDir..fileList[lineInd],  out)

         end

         ------------- output seq ------------
         for k = 1,  opt.output_nSeq do
            local out = image.load(data_path..'data/'..fileList[lineInd])
            --local out = dataMask:quick4Mask3Dforward(ori)
            -- out = image.rotate(out, 1.5)
            output_batch[k][batch_ind] = out:clone()

            lineInd = lineInd + selectStep
         end
         -- input_batch = input_batch:gt(0.1)
         -- output_batch = output_batch:gt(0.1)
         -- # TODO: make sure selectInd do not go out of bound[now mannuly?]
         selectIndStart = selectIndStart + 1 -- goto next minibatch
      end

      return input_batch, output_batch
   end

   -- dsample(20, 1, 64, 64)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                           local inputTable, targetTable = self:selectTensor(index)
                                           
                                           return {inputTable, targetTable}
                                       end})
   return datasetSeq

end
