require 'nn'

-- zero out diagonal values
local MakeDiagonalZero, parent = torch.class('nn.MakeDiagonalZero', 'nn.Module')

function MakeDiagonalZero:__init()
   parent.__init(self)
   self.gradInput = torch.Tensor()
end

function MakeDiagonalZero:updateOutput(input)
   -- input size batchsize x seqlen x seqlen
   assert(input:dim() == 3, 'invalid dimension')

   local batchsize = input:size(1)
   local seqlen = input:size(2)
   self.output:resizeAs(input):copy(input)

   for b = 1, batchsize do
      for s = 1, seqlen do
         self.output[{b,s,s}] = 0
      end
   end
   return self.output
end

function MakeDiagonalZero:updateGradInput(input, gradOutput)
   -- input size batchsize x seqlen x seqlen
   assert(input:dim() == 3, 'invalid dimension')
   
   local batchsize = input:size(1)
   local seqlen = input:size(2)

   self.gradInput:resizeAs(input):copy(gradOutput)

   for b = 1, batchsize do
      for s = 1, seqlen do
         self.gradInput[{b,s,s}] = 0
      end
   end

   return self.gradInput
end
