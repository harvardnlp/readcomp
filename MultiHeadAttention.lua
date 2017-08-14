-- Reference: https://arxiv.org/abs/1706.03762

local MultiHeadAttention, parent = torch.class('nn.MultiHeadAttention', 'nn.Sequential')
function MultiHeadAttention:__init(h, dmodel, dropout, mask_subsequent, single_input)
  -- expects input of size batchsize x seqlen x dmodel
  parent.__init(self)
  local dk = math.floor(dmodel / h)
  local dv = dk

  mask_subsequent = mask_subsequent and mask_subsequent or false
  single_input = single_input and single_input or true

  local multihead = nn.ConcatTable()
  for i = 1, h do
    local SQTK = nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.Bottle(nn.Linear(dmodel, dk))) -- Q : batchsize x seqlen x dk
        :add(nn.Sequential():add(nn.Bottle(nn.Linear(dmodel, dk))):add(nn.Transpose({2,3})))) -- K^T : batchsize x dk x seqlen
      :add(nn.MM()) -- Q * K^T : batchsize x seqlen x seqlen
      :add(nn.MulConstant(1 / math.sqrt(dk))) -- scaled dot-product attention
      :add(nn.Bottle(self:SoftMaxDropout(dropout)))

    if mask_subsequent == true then
      SQTK:add(nn.MaskSubsequentPositions())
    end

    local all = nn.Sequential()
    local all_input = single_input and nn.ConcatTable() or nn.ParallelTable()
    all_input:add(SQTK):add(nn.Bottle(nn.Linear(dmodel, dv))) -- V : batchsize x seqlen x dv
    all:add(all_input):add(nn.MM())

    multihead:add(all) -- (Q K^T) * V: batchsize x seqlen x dv
  end
  self:add(multihead) -- batchsize x seqlen x dmodel
      :add(nn.JoinTable(3)) -- batchsize x seqlen x (h * dv)
      :add(nn.Bottle(nn.Linear(h * dv, dmodel))) -- batchsize x seqlen x dmodel

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