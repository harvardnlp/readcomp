require 'nn'

-- zero out diagonal values
local MaskSubsequentPositions, parent = torch.class('nn.MaskSubsequentPositions', 'nn.Module')

function MaskSubsequentPositions:__init()
   parent.__init(self)
   self.gradInput = torch.Tensor()
end

function MaskSubsequentPositions:updateOutput(input)
   -- input size batchsize x seqlen x seqlen
   assert(input:dim() == 3, 'invalid dimension')

   local batchsize = input:size(1)
   local seqlen = input:size(2)

   self.output:resizeAs(input):copy(input)
   for i = 1, seqlen - 1 do
      self.output[{{}, i, {i + 1, seqlen}}] = -math.huge
   end
   return self.output
end

function MaskSubsequentPositions:updateGradInput(input, gradOutput)
   -- input size batchsize x seqlen x seqlen
   assert(input:dim() == 3, 'invalid dimension')
   
   local batchsize = input:size(1)
   local seqlen = input:size(2)

   self.gradInput:resizeAs(input):copy(gradOutput)
   for i = 1, seqlen - 1 do
      self.gradInput[{{}, i, {i + 1, seqlen}}] = 0
   end
   return self.gradInput
end
