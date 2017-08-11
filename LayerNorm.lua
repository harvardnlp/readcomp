-- Reference: https://arxiv.org/pdf/1607.06450.pdf (Section 3)

local LayerNorm, parent = torch.class('nn.LayerNorm', 'nn.Sequential')
function LayerNorm:__init(batchsize, hiddensize, eps, affine)
  -- expects input of size batchsize x seqlen x hiddensize

  parent.__init(self)
  eps = eps or 1e-10
  affine = (affine == nil) and true or affine

  self:add(nn.Transpose({1,2})) -- seqlen x batchsize x hiddensize
  local ln = nn.Sequential()
    :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(nn.Sequential():add(nn.Mean(1, 1)):add(nn.Replicate(hiddensize,2,1))))
    :add(nn.CSubTable()) -- x - mu
    :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(nn.Sequential() -- sqrt(sigma^2 + epsilon)
                    :add(nn.Power(2)):add(nn.Mean(1, 1))
                    :add(nn.AddConstant(eps)):add(nn.Sqrt())
                    :add(nn.Replicate(hiddensize,2,1))))
    :add(nn.CDivTable())

  self:add(nn.Bottle(ln))

  if affine then
    self:add(nn.SplitTable(1))
        :add(nn.MapTable()
                :add(nn.Sequential()
                        :add(nn.CMul(batchsize, hiddensize))
                        :add(nn.CAdd(batchsize, hiddensize))
                        :add(nn.Unsqueeze(1))))
        :add(nn.JoinTable(1))
  end
  self:add(nn.Transpose({1,2}))
end