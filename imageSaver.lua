-- imageSaver.lua
--[[
to make life easier
this function can save table or tensor with dimension more than 3
specially, 
- it can save weight tensor with multidepth as one image
- it can save flow image in [H, W, D=2] with colorwheel indicated the direction of the flow and the magitude indicated by the lightness of the image
]]