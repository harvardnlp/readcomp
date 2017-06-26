require 'nn'

local ZeroToNegInf, parent = torch.class('nn.ZeroToNegInf', 'nn.Module')

-- convert all zero values to -inf, this layer should be used before softmax()
-- to avoid attention on padded values
function ZeroToNegInf:__init()
   parent.__init(self)
end

function ZeroToNegInf:updateOutput(input)
   -- input is batch x seqlen
   self.output:resizeAs(input):copy(input)
   local batch = input:size(1)
   for i = 1, batch do
      if input[i]:ne(0):sum() > 0 then
         self.output[i][input[i]:eq(0)] = torch.log(0)
      end
   end
   return self.output
end

function ZeroToNegInf:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(gradOutput)
   self.gradInput[input:eq(0)] = 0

   return self.gradInput
end