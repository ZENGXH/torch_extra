require 'image'

local data_verbose = false

if opt.appliedMask and dataMask == nil then
   require '../torch_extra/dataMask'
   dataMask = dataMask(opt.imageH, opt.imageW, opt.maskStride, opt.maskStride * opt.maskStride) 
end


function getdataSeq_hko(mode, data_path)
   local mode = mode or "train"
   local data_path = data_path or '../helper/'
   -- local data = torch.DiskFile(datafile,'r'):readObject()
   -- local data_path = 
   -- data size (totalInstances or nsamples=2000?, sequence_length=20, 1, 64, 64)
   local datasetSeq ={}
   gpuflag = opt.gpuflag or false

   if gpuflag then
      print('load data to gpu')
      require 'cunn'
      typeT = torch.Tensor():cuda()
   else
      typeT = torch.Tensor():float()
   end
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
   end
   assert(table.getn(fileList) == nseq * nsamples)
   local ids = torch.range(1, nseq * nsamples)
   local seqHeads = torch.Tensor(nsamples)

   if mode == 'train' then
      print('training mode, shffule heads')
      shuffleID = torch.randperm(nsamples) 
      -- range from 1 to numOfBatches, for shuffle purpose
   else
      shuffleID = torch.range(1, nsamples)
      -- then shuffleBatchID[i] = i
   end   

   for i = 1, nsamples do
      seqHeads[i] = (shuffleID[i] - 1) * nseq + 1 
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
            --local out = dataMask:quick4Mask3Dforward(ori)
            input_batch[batch_ind][k] = out:clone()
            lineInd = lineInd + selectStep
         end
         ------------- output seq ------------
         for k = 1,  opt.output_nSeq do
            local out = image.load(data_path..'data/'..fileList[lineInd])
            --local out = dataMask:quick4Mask3Dforward(ori)
            output_batch[batch_ind][k] = out:clone()
            lineInd = lineInd + selectStep
         end
         -- # TODO: make sure selectInd do not go out of bound[now mannuly?]
         selectIndStart = selectIndStart + 1 -- goto next minibatch
      end

      ------------- make it a table -----------
      for k = 1, opt.input_nSeq do
         if gpuflag then
            table.insert(inputTable, input_batch[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW):cuda())
         else
            table.insert(inputTable, input_batch[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW))
         end
      end

       for k = 1, opt.output_nSeq do
         if gpuflag then
            table.insert(targetTable, output_batch[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW):cuda())
         else
            table.insert(targetTable, output_batch[{{}, {i}, {}, {}, {}}]:select(2,1):reshape(opt.batchSize, opt.imageDepth, opt.imageH, opt.imageW))
         end
      end     

      return inputTable, targetTable
   end

   -- dsample(20, 1, 64, 64)
   
   setmetatable(datasetSeq, {__index = function(self, index)
                                           local inputTable, targetTable = self:selectSeq(index)
                                           
                                           return {inputTable, targetTable}
                                       end})
   return datasetSeq
end
