-- test_imageScale.lua

require 'image'
require 'torch'
require 'nn'
dofile '../torch_extra/imageScale.lua'
i = image.lena()
i = image.rgb2y(i)
net = nn.imageScale(i:size(2)/2, i:size(3)/2, 'bicubic')
net2 = nn.imageScale(i:size(2)*2, i:size(3)*2, 'bicubic')
image.display(i)
print('input tensor')
image.display(net:forward(i))
image.display(net2:forward(i))

print('input table')

i = image.lena()
t = {i, i}
out = net:forward(t)
print(out)