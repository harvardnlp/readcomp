require 'nn'
require 'KMaxFilter'
require 'MakeDiagonalZero'

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1.1e-4

function TestKMaxFilter()
	local ntests = 100
	local max_dim1 = 1024
	local max_dim2 = 1024

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

jac = nn.Jacobian
sjac = nn.SparseJacobian

TestKMaxFilter()