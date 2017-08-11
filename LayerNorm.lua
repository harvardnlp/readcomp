-- Reference: https://arxiv.org/pdf/1607.06450.pdf (Section 3)

local LayerNorm, parent = torch.class('nn.LayerNorm', 'nn.Sequential')
function LayerNorm:__init(batchsize, hiddensize, eps, affine)
   -- expects input of size batchsize x hiddensize

   parent.__init(self)
   eps = eps or 1e-10
   affine = (affine == nil) and true or affine

   self:add(nn.ConcatTable()
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

   if affine then
      self:add(nn.CMul(batchsize, hiddensize))
      self:add(nn.CAdd(batchsize, hiddensize))
   end
end