require 'nn'
require 'CAddTableBroadcast'

x1 = torch.range(1,12):view(2,3,2)
x2 = torch.range(1,6):view(3,2)

y = torch.DoubleTensor({56,62})

model = nn.Sequential()
	:add(nn.CAddTableBroadcast())
	:add(nn.Sum(1))
	:add(nn.Sum(1))

crit = nn.MSECriterion()

function foba(x)
	o = model:forward({x1,x})

	loss = crit:forward(o,y)
	dldo = crit:backward(o,y)

	dldx = model:backward({x1,x},dldo)
	return dldx, loss
end

dldx, _ = foba(x2)

step = 1e-6
for i = 1, x2:size(1) do
	for j = 1, x2:size(2) do
		xupper = x2:clone()
		xupper[i][j] = xupper[i][j] + step

		_, loss_upper = foba(xupper)

		xlower = x2:clone()
		xlower[i][j] = xlower[i][j] - step

		_, loss_lower = foba(xlower)

		numeric_grad = (loss_upper - loss_lower) / (2 * step)

		print(torch.abs(numeric_grad - dldx[2][i][j]))
	end
end

