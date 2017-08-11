-- Reference: https://arxiv.org/abs/1706.03762

local MultiHeadAttention, parent = torch.class('nn.MultiHeadAttention', 'nn.Sequential')
function MultiHeadAttention:__init(h, dmodel, dropout)
  -- expects input of size batchsize x seqlen x dmodel
  parent.__init(self)
  local dk = math.floor(dmodel / h)
  local dv = dk

  local multihead = nn.ConcatTable()
  for i = 1, h do
    multihead:add(nn.Sequential()
          :add(nn.ConcatTable()
            :add(nn.Sequential()
              :add(nn.ConcatTable()
                :add(nn.Bottle(nn.MaskZero(nn.Linear(dmodel, dk), 1))) -- Q : batchsize x seqlen x dk
                :add(nn.Sequential():add(nn.Bottle(nn.MaskZero(nn.Linear(dmodel, dk), 1))):add(nn.Transpose({2,3})))) -- K^T : batchsize x dk x seqlen
              :add(nn.MM()) -- Q * K^T : batchsize x seqlen x seqlen
              :add(nn.MulConstant(1 / math.sqrt(dk))) -- scaled dot-product attention
              :add(nn.Bottle(nn.MaskZero(self:SoftMaxDropout(dropout), 1))))
            :add(nn.Bottle(nn.MaskZero(nn.Linear(dmodel, dv),1)))) -- V : batchsize x seqlen x dv
          :add(nn.MM())) -- (Q K^T) * V: batchsize x seqlen x dv
  end
  self:add(multihead) -- batchsize x seqlen x dmodel
      :add(nn.JoinTable(3)) -- batchsize x seqlen x (h * dv)
      :add(nn.Bottle(nn.MaskZero(nn.Linear(h * dv, dmodel), 1))) -- batchsize x seqlen x dmodel

  if dropout > 0 then
    self:add(nn.Dropout(dropout))
  end
end

function MultiHeadAttention:SoftMaxDropout(dropout)
  if dropout > 0 then
    return nn.Sequential():add(nn.SoftMax()):add(nn.Dropout(dropout))
  else
    return nn.SoftMax()
  end
end