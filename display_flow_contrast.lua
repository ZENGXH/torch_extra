require 'torch'
require 'image'

--torch.setdefaulttensortype('torch.FloatTensor')

function flow2colour(flow_in)     -- bdhw
  -- flow[batchSize, depth(2), h, w]
  local flow = flow_in:clone()
  flow = flow:float()

  if flow:nDimension() == 3 then -- batchsize = 1, generate more than one image
    local img = flow2colour3D(flow) 
  else -- batchsize > 1
    local numsOfBatch = flow:size(1)
    local img = torch.Tensor(3, flow:size(3), flow:size(4), numsOfBatch)  -- 3 == colorwheel:size(2)
    for n = 1, numsOfBatch do
        local subflow = flow:select(1, n)
        local subimg = img:select(4, n) 
        subimg = flow2colour3D(subflow)
    end
    img:resize(3, flow:size(3), flow:size(4) * numsOfBatch) -- concate flow for different batch
  end
  img = img:float()
  return img
end

function flow2colour3D(flow)    
  -- print(flow)
  assert(flow:nDimension() == 3)
  nBands = flow:size(1)
  height = flow:size(2)
  width  = flow:size(3) 

  assert(nBands == 2)    
  if not opt.gpuflag then
    u = flow[{{1},{},{}}]:float():clone():squeeze()
    v = flow[{{2},{},{}}]:float():clone():squeeze()
  else
    u = flow[{{1},{},{}}]:double():clone():squeeze()
    v = flow[{{2},{},{}}]:double():clone():squeeze()    
  end
  --print (#u)
  --print (#v)

  maxu = u:max()
  minu = u:min()

  maxv = v:max()
  minv = v:min()

  rad = torch.sqrt(torch.pow(u, 2) + torch.pow(v, 2)) 
  maxrad = rad:max()

  print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n', maxrad, minu, maxu, minv, maxv)
  eps = 1e-6
  u = u / (maxrad + eps);
  v = v / (maxrad + eps);

  -- compute color

  img = computeColor(u, v) 
    
  return img
end

function displayFlowTest()
  truerange = 1
  height    = 151
  width     = 151
  rg = truerange * 1.04

  s2 = torch.round(height / 2)

  x,y = meshgrid(width, height)

  u = x * rg / s2 - rg
  v = y * rg / s2 - rg

  img = computeColor(u / truerange, v / truerange)

  img[{{},{s2},{}}]:fill(0)
  img[{{},{},{s2}}]:fill(0)

  image.display(img)
end

function computeColor(u,v)
  colorwheel = makeColorWheel()
  --print (colorwheel)
  ncols = colorwheel:size(1)

  rad = torch.sqrt(torch.pow(u,2) + torch.pow(v,2)):typeAs(u)

  a = torch.Tensor():typeAs(u)

  a:resizeAs(u)
  for i = 1, u:size(1) do
    for j = 1, u:size(2) do       
      a[i][j] = math.atan2(-v[i][j], -u[i][j]) / math.pi  
      -- math.atan2(x, y) == arc tangent of x/y (in radians)
      -- math.atan2(x, y) = thelta/pi => tan(thelta/pi) = x / y
      -- 
    end
  end

  fk = (a + 1) / 2 * (ncols - 1) + 1  -- many cols; range [1, ncols]
   
  k0 = torch.floor(fk)  -- k0: fk truncate to int; {1, 2, ..., ncols}

  k1 = k0 + 1

  for i = 1, k1:size(1) do
    for j = 1, k1:size(2) do
      if k1[i][j] == ncols + 1 then
        k1[i][j] = 1
      end
    end
  end

  f = fk - k0  

  ch = colorwheel:size(2) -- 3

  img = torch.Tensor(ch, u:size(1), u:size(2))

  for i = 1, colorwheel:size(2) do -- i = 1:3

    tmp = colorwheel[{{},{i}}]--:clone(), tmp [ncols, 3]

    local col0 = torch.Tensor():typeAs(u)
    col0:resizeAs(u):zero()
    local col1 = torch.Tensor():typeAs(u)
    col1:resizeAs(u):zero()
    
    for i1=1,col0:size(1) do
      for i2 = 1,col0:size(2) do 
        --if k0[i1][i2]>0 then 
          col0[i1][i2] = tmp[k0[i1][i2]] / 255 -- #col0 = k0 = floor(fk)
        --end
        --if k1[i1][i2]>0 then 
          col1[i1][i2] = tmp[k1[i1][i2]] / 255 -- #col1 = k1 = k0 + 1 = floor(fk) + 1, (if k1 == ncols then 1)
        --end
      end
    end
    f1 = f:clone()        -- f1 = f = fk - k0 = fk - floor(fk); range[0, 1]
    f1:neg():add(1)
    f2 = f:clone()        -- f2 = -f + 1 = 1 - (fk - floor(fk)); range[0, 1]

    tmm1 = f1:cmul(col0)  -- tmm1 = f1 * #col0 = f1 * floor(fk)       = (fk - floor(fk)) * floor(fk)
    tmm2 = f2:cmul(col1)  -- tmm2 = f2 * #col1 = f2 * (floor(fk) + 1) = [1 - (fk - floor(fk))] * (floor(fk) + 1)
    col = tmm1:add(tmm2)  -- col = tmm1 + tmm2 
                          --     = (fk - floor(fk)) * floor(fk) + [1 - (fk - floor(fk))] * (floor(fk) + 1)
                          -- if fk = floor(fk), then col = floor(fk + 1)
                          -- if fk = floor(fk) + 0.5, then col = 0.5 * floor(fk) + 0.5 * (floot(fk))
                          -- fk in range [1, ncols], col in range range [1, ncols]
    idx = torch.Tensor():typeAs(rad)
    idx:resizeAs(rad)
    for i1 = 1,idx:size(1) do
      for i2 = 1,idx:size(2) do 
        if rad[i1][i2] <= 1 then
          col[i1][i2] = 1 - rad[i1][i2] * (1 - col[i1][i2])  -- color = 1 + (color - 1) * rad
          -- color = 1 - magOfFlow + color * magOfFlow
          -- rad(radius) is the magitude of the flow, represent by the intensity of the color 
          -- if rad = 1, color = color
          -- if rad = 0, color = 1
          -- if rad = 0.5, color = 1 - 0.5 + color * 0.5
        else -- rad > 1, 
          col[i1][i2] = col[i1][i2] * 0.75
        end
      end
    end
    
    img[{{i},{},{}}] = torch.floor(col * 255):clone()       
  end
  return img
end   


function makeColorWheel()
  local RY = 15
  local YG = 6
  local GC = 4
  local CB = 11
  local BM = 13
  local MR = 6

  local ncols = RY + YG + GC + CB + BM + MR

  local colorwheel = torch.Tensor(ncols, 3):zero()

  local col = 0;
  --RY
  colorwheel[{{1, RY}, {1}}]:fill(255)
  local tmp1 = range(0, RY-1):t()
  colorwheel[{{1, RY}, {2}}]:copy(torch.floor(tmp1 * 255 / RY))
  col = col + RY

  --YG
  local tmp2 = range(0, YG-1):t()
  colorwheel[{{col + 1, col + YG}, {1}}]:copy(-torch.floor(tmp2 * 255 / YG) + 255)
  colorwheel[{{col + 1, col + YG}, {2}}]:fill(255)
  col = col + YG

  --GC
  local tmp3 = range(0, GC - 1):t()
  colorwheel[{{col + 1, col + GC}, {2}}]:fill(255)
  colorwheel[{{col + 1, col + GC}, {3}}]:copy(torch.floor(tmp3 * 255 / GC))
  col = col + GC

  --CB
  local tmp4 = range(0, CB - 1):t()
  colorwheel[{{col + 1, col + CB}, {2}}]:copy(-torch.floor(tmp4 * 255 / CB) + 255)
  colorwheel[{{col + 1, col + CB}, {3}}]:fill(255)
  col = col + CB

  --BM
  local tmp5 = range(0, BM - 1):t()
  colorwheel[{{col + 1, col + BM}, {3}}]:fill(255)
  colorwheel[{{col + 1, col + BM}, {1}}]:copy(torch.floor(tmp5 * 255 / BM))
  col = col + BM

  --MR
  local tmp6 = range(0, MR - 1):t()
  colorwheel[{{col + 1, col + MR}, {3}}]:copy(-torch.floor(tmp6 * 255 / MR) + 255)
  colorwheel[{{col + 1, col + MR}, {1}}]:fill(255)
  
  collectgarbage()
  return colorwheel
end

function range(a,b)
  r = torch.Tensor(1, b - a + 1):zero()
  for i=a, b do
    r[1][i - a + 1] = i
  end
  return r
end

function meshgrid(h, w)
  x = torch.Tensor(h, w):zero()
  y = torch.Tensor(h, w):zero()
  for i=1, h do
    for j=1, w do
      x[i][j] = j
      y[i][j] = i
    end
  end
  return x, y
end

--displayFlowTest()