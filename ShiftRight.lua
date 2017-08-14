require 'nn'

-- zero out diagonal values
local ShiftRight, parent = torch.class('nn.ShiftRight', 'nn.Module')

function ShiftRight:__init()
   parent.__init(self)
   self.gradInput = torch.Tensor()
end

function ShiftRight:updateOutput(input)
   -- input size batchsize x seqlen

   local seqlen = input:size(2)
   self.output:resizeAs(input):zero()
   self.output[{{}, {2, seqlen}}] = input[{{}, {1, seqlen - 1}}]
   return self.output
end

function ShiftRight:updateGradInput(input, gradOutput)
   -- input size batchsize x seqlen
   
   local seqlen = input:size(2)
   self.gradInput:resizeAs(input):zero()
   self.gradInput[{{}, {1, seqlen - 1}}] = gradOutput[{{}, {2, seqlen}}]
   return self.gradInput
end
