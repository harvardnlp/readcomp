require 'nn'

-- zero out diagonal values
local Triufy, parent = torch.class('nn.Triufy', 'nn.Module')

function Triufy:__init()
   parent.__init(self)
   self.gradInput = torch.Tensor()
end

function Triufy:updateOutput(input)
   -- input size batchsize x seqlen x seqlen
   self.output:resizeAs(input):copy(input)
   local batchsize = input:size(1)
   local seqlen = input:size(2)
   for b = 1, batchsize do
      for s = 2, seqlen do
         self.output[{{}, s, {1, s - 1}}]:zero()
      end
   end
   return self.output
end

function Triufy:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput)
   local batchsize = input:size(1)
   local seqlen = input:size(2)
   for b = 1, batchsize do
      for s = 2, seqlen do
         self.gradInput[{{}, s, {1, s - 1}}]:zero()
      end
   end
   return self.gradInput
end
