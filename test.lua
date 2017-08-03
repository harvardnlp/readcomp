require 'nn'
require 'KMaxFilter'
require 'MakeDiagonalZero'
require 'MaskZeroSeqBRNNFinal'
require 'MaxNodeMarginal'

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1.1e-4

local nntest = torch.TestSuite()

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

nn.test{'KMaxFilter', 'MakeDiagonalZero', 'MaskZeroSeqBRNNFinal', 'MaxNodeMarginal'}