require 'nn'

-- extract non-zero final states of BRNN when inputs contain zeros
-- output = concat( final non-zero state of forward(x), final non-zero state of backward(x) )
local MaskExtraction, parent = torch.class('nn.MaskExtraction', 'nn.Module')

function MaskExtraction:__init(maskValue, numExtract)
   parent.__init(self)
   self.gradInput = {}
   self.maskValue = maskValue
   self.numExtract = numExtract
end

function MaskExtraction:updateOutput(input)
  -- data size batchsize x seqlen x 2H
  local data = input[1]
  local mask = input[2]
  assert(data:dim() == 3, 'invalid dimension')

  local batchsize = data:size(1)
  local seqlen = data:size(2)
  local hidsize = data:size(3)
  self.output:resize(batchsize, self.numExtract, hidsize):zero()

  for b = 1, batchsize do
    local extracted = 1
    for s = 1, seqlen do
      if mask[b][s] == self.maskValue then
        self.output[b][extracted] = data[b][s]
        if extracted == self.numExtract then
          break
        end
        extracted = extracted + 1
      end
    end
  end
  return self.output
end

function MaskExtraction:updateGradInput(input, gradOutput)
  -- data size batchsize x seqlen x 2H
  local data = input[1]
  local mask = input[2]
  assert(data:dim() == 3, 'invalid dimension')
   
  local batchsize = data:size(1)
  local seqlen = data:size(2)
  local hidsize = data:size(3)

  self.gradInput[1] = self.gradInput[1] or data.new()
  self.gradInput[2] = self.gradInput[2] or mask.new()

  self.gradInput[1]:resizeAs(data):zero()
  self.gradInput[2]:resizeAs(mask):zero()

  for b = 1, batchsize do
    local extracted = 1
    for s = 1, seqlen do
      if mask[b][s] == self.maskValue then
        self.gradInput[1][b][s] = gradOutput[b][extracted]
        if extracted == self.numExtract then
          break
        end
        extracted = extracted + 1
      end
    end
  end

  return self.gradInput
end
