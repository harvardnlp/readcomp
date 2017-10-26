require 'nn'

local MaskZeroAttention, parent = torch.class('nn.MaskZeroAttention', 'nn.Module')

function MaskZeroAttention:__init()
   parent.__init(self)
end

function MaskZeroAttention:updateOutput(input)
  -- data size batchsize x seqlen
  self.output:resizeAs(input):copy(input)
  self.output[input:eq(0)] = -math.huge
  return self.output
end

function MaskZeroAttention:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput[input:eq(0)] = 0
  return self.gradInput
end
