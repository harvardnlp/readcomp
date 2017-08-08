-- Reference: https://arxiv.org/abs/1706.03762

local PositionWiseFFNN, parent = torch.class('nn.PositionWiseFFNN', 'nn.Sequential')
function PositionWiseFFNN:__init(hidsize, dff, dropout)
  -- expects input of size batchsize x seqlen x hidsize
  parent.__init(self)

  self:add(nn.Bottle(nn.Sequential()
      :add(nn.Linear(hidsize, dff))
      :add(nn.ReLU())
      :add(nn.Linear(dff, hidsize))))

  if dropout > 0 then
    self:add(nn.Dropout(dropout))
  end
end
