require 'nn'
require 'rnn'
require 'ShiftRight'
require 'KMaxFilter'
require 'MakeDiagonalZero'
require 'MaskZeroSeqBRNNFinal'
require 'MaxNodeMarginal'
require 'SinusoidPositionEncoding'
require 'MultiHeadAttention'
require 'PositionWiseFFNN'
require 'LayerNorm'
require 'MaskExtraction'
require 'MaskZeroAttention'


cmd = torch.CmdLine()
cmd:text()
cmd:option('--maskzero', false, 'enable unit tests for maskzero')

cmd:text()
local opt = cmd:parse(arg or {})

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1.1e-4

local nntest = torch.TestSuite()

function nntest.ShiftRight()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(2, max_dim2)
    local dim3 = math.random(1, max_dim2)

    local module = nn.ShiftRight()

    local input = torch.rand(dim1,dim2,dim3):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.KMaxFilter()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)

    local k = math.random(1, dim2)
    local module = nn.KMaxFilter(k)

      -- 3D
    local input = torch.rand(dim1,dim2,dim2):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.MakeDiagonalZero()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)

    local module = nn.MakeDiagonalZero()

      -- 3D
    local input = torch.rand(dim1,dim2,dim2):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.MaskZeroSeqBRNNFinal()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3)

    local module = nn.MaskZeroSeqBRNNFinal()

      -- 3D
    local input = torch.rand(dim1,dim2,dim3 * 2):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.MaxNodeMarginal()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3)

    local module = nn.MaxNodeMarginal()

      -- 3D
    local input = torch.rand(dim1,dim2,dim3):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.SinusoidPositionEncoding()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16

  if opt.maskzero then
    local x = torch.rand(8, 32, 16)
    x[{{}, {1,2}}]:zero()
    x[{{}, {10,11}}]:zero()

    local y = nn.SinusoidPositionEncoding(1024, 16):forward(x)
    mytester:assertlt(y[{{}, 1}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{}, 2}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},10}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},11}]:ne(0):sum(),precision, 'error on maskzero ')
  end

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3)

    local module = nn.SinusoidPositionEncoding(1024, dim3)

    -- 3D
    local input = torch.rand(dim1,dim2,dim3):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.MultiHeadAttention()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16

  if opt.maskzero then
    local x = torch.rand(8, 32, 16)
    x[{{}, {1,2}}]:zero()
    x[{{}, {10,11}}]:zero()

    local y = nn.MultiHeadAttention(8, 16, 0.1,false,true):forward(x)
    mytester:assertlt(y[{{}, 1}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{}, 2}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},10}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},11}]:ne(0):sum(),precision, 'error on maskzero ')
  end

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3)
    local h    = math.random(1, dim3)

    local module = nn.MultiHeadAttention(h, dim3, 0)

    local input = torch.rand(dim1,dim2,dim3):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.PositionWiseFFNN()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16
  local max_dff  = 16

  if opt.maskzero then
    local x = torch.rand(8, 32, 16)
    x[{{}, {1,2}}]:zero()
    x[{{}, {10,11}}]:zero()

    local y = nn.PositionWiseFFNN(16, 3, 0.1):forward(x)
    mytester:assertlt(y[{{}, 1}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{}, 2}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},10}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},11}]:ne(0):sum(),precision, 'error on maskzero ')
  end

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3)
    local dff  = math.random(1, max_dff)

    local module = nn.PositionWiseFFNN(dim3, dff, 0)

    local input = torch.rand(dim1,dim2,dim3):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function ln(x)
  xmu = x - x:mean(2):expand(2,3)
  sigma = torch.sqrt(torch.mean(torch.cmul(xmu, xmu), 2):expand(2,3) + 1e-10)
  return torch.cdiv(xmu, sigma)
end

function nntest.LayerNorm()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 16

  if opt.maskzero then
    local x = torch.rand(8, 32, 16)
    x[{{}, {1,2}}]:zero()
    x[{{}, {10,11}}]:zero()

    local y = nn.LayerNorm(8, 16, 1e-10, true):forward(x)
    mytester:assertlt(y[{{}, 1}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{}, 2}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},10}]:ne(0):sum(),precision, 'error on maskzero ')
    mytester:assertlt(y[{{},11}]:ne(0):sum(),precision, 'error on maskzero ')
  end

  local x = torch.rand(5,2,3)
  z = nn.LayerNorm(5, 3, 1e-10, false):forward(x)
  for i = 1, x:size(1) do
    mytester:assertlt(torch.sum(torch.abs(z[i] - ln(x[i]))),precision, 'error on manual test ')
  end
  nn.LayerNorm(5, 3, 1e-10, true):forward(x)

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3)

    local module = nn.LayerNorm(dim1, dim3, 1e-10, t % 2 == 0)

    local input = torch.rand(dim1,dim2,dim3):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

function nntest.MaskExtraction()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16
  local max_dim3 = 8

  local x = torch.rand(6, 32, 16) -- batchsize x seqlen x hidsize
  local m = torch.ones(6, 32)
  m[1][4] = 2
  m[1][13] = 2
  m[1][22] = 2
  m[{2,{10,21}}] = 2
  m[3][5] = 2
  m[3][10] = 2
  m[4][1] = 2
  m[5][32] = 2

  local y = nn.MaskExtraction(2, 5):forward({x,m})
  mytester:assertlt(y[1][1][{{1,8}}]:ne(x[1][3][{{1,8}}]):sum(),precision, 'error')
  mytester:assertlt(y[1][1][{{9,16}}]:ne(x[1][5][{{9,16}}]):sum(),precision, 'error')

  mytester:assertlt(y[1][2][{{1,8}}]:ne(x[1][12][{{1,8}}]):sum(),precision, 'error')
  mytester:assertlt(y[1][2][{{9,16}}]:ne(x[1][14][{{9,16}}]):sum(),precision, 'error')

  mytester:assertlt(y[1][3][{{1,8}}]:ne(x[1][21][{{1,8}}]):sum(),precision, 'error')
  mytester:assertlt(y[1][3][{{9,16}}]:ne(x[1][23][{{9,16}}]):sum(),precision, 'error')

  for i = 10, 14 do
    mytester:assertlt(y[{2,i-9,{1,8}}]:ne(x[{2,i-1,{1,8}}]):sum(),precision, 'error')
    mytester:assertlt(y[{2,i-9,{9,16}}]:ne(x[{2,i+1,{9,16}}]):sum(),precision, 'error')
  end

  mytester:assertlt(y[3][1][{{1,8}}]:ne(x[3][4][{{1,8}}]):sum(),precision, 'error')
  mytester:assertlt(y[3][1][{{9,16}}]:ne(x[3][6][{{9,16}}]):sum(),precision, 'error')

  mytester:assertlt(y[3][2][{{1,8}}]:ne(x[3][9][{{1,8}}]):sum(),precision, 'error')
  mytester:assertlt(y[3][2][{{9,16}}]:ne(x[3][11][{{9,16}}]):sum(),precision, 'error')

  mytester:assertlt(y[4][1][{{1,8}}]:abs():sum(),precision, 'error')
  mytester:assertlt(y[4][1][{{9,16}}]:ne(x[4][2][{{9,16}}]):sum(),precision, 'error')

  mytester:assertlt(y[5][1][{{1,8}}]:ne(x[5][31][{{1,8}}]):sum(),precision, 'error')
  mytester:assertlt(y[5][1][{{9,16}}]:abs():sum(),precision, 'error')

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)
    local dim3 = math.random(1, max_dim3) * 2

    local input = torch.rand(dim1,dim2,dim3)
    local mask = torch.rand(dim1,dim2):mul(2):ceil()
    local extendedInput = torch.zeros(dim1*dim2*dim3 + dim1*dim2)
    extendedInput[{{1, dim1*dim2*dim3}}] = input
    extendedInput[{{dim1*dim2*dim3 + 1, dim1*dim2*dim3 + dim1*dim2}}] = mask

    local module = nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Sequential():add(nn.Narrow(1, 1, dim1 * dim2 * dim3)):add(nn.View(dim1, dim2, dim3)))
        :add(nn.Sequential():add(nn.Narrow(1, dim1 * dim2 * dim3 + 1, dim1 * dim2)):add(nn.View(dim1, dim2))))
      :add(nn.MaskExtraction(2, 5))
      :add(nn.Sum())

    local jac_fprop = nn.Jacobian.forward(module, extendedInput, extendedInput, perturbation)
    local jac_bprop = nn.Jacobian.backward(module, extendedInput)
    local error = jac_fprop-jac_bprop
    local err = error:abs():max()

    mytester:assertlt(err, precision, 'error on state ')

    -- IO
    local ferr,berr = jac.testIO(module,extendedInput)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end


function nntest.MaskZeroAttention()
  local ntests = 5
  local max_dim1 = 8
  local max_dim2 = 16

  for t = 1, ntests do
    local dim1 = math.random(1, max_dim1)
    local dim2 = math.random(1, max_dim2)

    local module = nn.MaskZeroAttention()

    local input = torch.rand(dim1,dim2):zero()
    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')

    local ferr,berr = jac.testIO(module,input)
    mytester:eq(ferr, 0, torch.typename(module) .. ' - i/o forward err ', precision)
    mytester:eq(berr, 0, torch.typename(module) .. ' - i/o backward err ', precision)
  end
end

mytester:add(nntest)

jac = nn.Jacobian
sjac = nn.SparseJacobian

function nn.test(tests, seed)
   -- Limit number of threads since everything is small
   local nThreads = torch.getnumthreads()
   torch.setnumthreads(1)
   -- randomize stuff
   local seed = seed or (1e5 * torch.tic())
   print('Seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   mytester:run(tests)
   torch.setnumthreads(nThreads)
   return mytester
end

nn.test{
  'ShiftRight',
  'KMaxFilter', 
  'MakeDiagonalZero', 
  'MaskZeroSeqBRNNFinal', 
  'MaxNodeMarginal', 
  'SinusoidPositionEncoding', 
  'MultiHeadAttention', 
  'LayerNorm', 
  'PositionWiseFFNN',
  'MaskExtraction',
  'MaskZeroAttention'
}