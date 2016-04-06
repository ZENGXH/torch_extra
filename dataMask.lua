require 'torch'
require 'image'

local dataMask = torch.class('dataMask')


function dataMask:__init(height, width, stride, numOfMask) 
    print('dataMake formed')
    self.stride = stride or 2
    self.height = height or 100
    self.width = width or 100
    self.numOfMask = numOfMask or 4
    self.maskList = {}
    self.subheight = self.height / self.stride
    self.subwidth = self.width / self.stride
    self.gpuFlag = gpuFlag or false
    
    if self.numOfMask == 4 then
        local i = 1
        for offsetx = 0,1 do
            for offsety = 0,1 do
                self.maskList[i] = self:createMask(offsetx, offsety) 
                i = i + 1
            end
        end
    end
    -- self:test()
end

function dataMask:createMask(offsetx, offsety) 
    local offsetx = offsetx or 0
    local offsety = offsety or 0
    local mask = torch.ByteTensor(self.height, self.width):fill(0)
    for i = 1, self.subheight do
       for j = 1, self.subheight do 
           mask[self.stride * i - 1 + offsety][self.stride * j - 1 + offsetx] = 1
       end
    end
    return mask
end

function dataMask:applyMask(image, mask)
    if self.numOfMask == 1 then
      return image
    end

    assert(image:nDimension() == 3)
    local height = image:size(2)
    local width = image:size(3)
    assert(height == self.height)
    assert(width == self.width)
    
    local img = image:clone():float()
    local subImg = torch.Tensor(self.subheight, self.subwidth):fill(0):float()
    local newimg = img:maskedSelect(mask):clone():resizeAs(subImg)
    return newimg
end

function dataMask:recoverMask(originImg, mask, subimage)
    if self.numOfMask == 1 then
      return originImg
    end

    local subimg = subimage:clone():float()
    -- local img = torch.Tensor(height, width):fill(0):float()
    originImg:maskedCopy(mask, subimg)
    return originImg
end


function dataMask:quick4Mask3Dforward(originImg)
    if self.numOfMask == 1 then
      return originImg
    end
   assert(torch.isTensor(originImg))
   assert(originImg:nDimension() == 2 or originImg:nDimension() == 3)
   if originImg:nDimension() == 2 then
       image:resize(1, originImg:size(1), originImg:size(2))
   end

   assert( self.height == originImg:size(2))
   assert( self.width == originImg:size(3))

   self.imageFakeDepth = 4 -- not really depth, can be regard as batchsize
   local newImage = torch.Tensor(self.imageFakeDepth, self.subheight, self.subwidth):float()

   for i = 1, 4 do

           local mask = self.maskList[i]
           local newSubImage = self:applyMask(originImg, mask)
           newImage[i] = newSubImage
    end
  return newImage
end

function dataMask:quick4Mask3Dbackward(subImage_in)
--   print('backward image size:')
   -- print(subImage_in:size())   
   assert(subImage_in:nDimension() >= 3)
   local subImage = subImage_in:clone()
   if(subImage:nDimension() == 4) then
        subImage = subImage:squeeze()
  end
   -- sub image in size(4, height, width)

   assert(self.imageFakeDepth == subImage:size(1)) -- not really depth, can be regard as batchsize
   local originImg = torch.Tensor(1, self.height, self.width):float()
   for i = 1, 4 do
        self:recoverMask(originImg, self.maskList[i], subImage[i])
    end
  return originImg
end


function dataMask:test()
   print('please test with qlua compiler')
   local im = image.lena()
   im = image.rgb2y(im)
   local newImg = self:quick4Mask3Dforward(im)
   for i = 1, 4 do
      local single_subImg_fordisplay = newImg:select(1, i) 
      image.display(single_subImg_fordisplay)
   end

   print('shffule back: ')
   local originImg = self:quick4Mask3Dbackward(newImg) 
   image.display(originImg)
   print('check if pass test')
end


