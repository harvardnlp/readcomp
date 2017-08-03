require 'nn'

-- extract non-zero final states of BRNN when inputs contain zeros
-- output = concat( final non-zero state of forward(x), final non-zero state of backward(x) )
local KMaxFilter, parent = torch.class('nn.KMaxFilter', 'nn.Module')

function KMaxFilter:__init(k)
   parent.__init(self)
   self.gradInput = torch.Tensor()
   self.k = k
end

function KMaxFilter:updateOutput(input)
   -- input size batchsize x seqlen x seqlen
   assert(input:dim() == 3, 'invalid dimension')

   local batchsize = input:size(1)
   local seqlen = input:size(2)
   self.output:resizeAs(input):zero()

   for b = 1, batchsize do
      for s = 1, seqlen do
         local kv, ki = input[b][s]:topk(self.k, true)
         for i = 1, ki:size(1) do
            self.output[{b,s,ki[i]}] = input[{b,s,ki[i]}]
         end
      end
   end
   return self.output
end

function KMaxFilter:updateGradInput(input, gradOutput)
   -- input size batchsize x seqlen x seqlen
   assert(input:dim() == 3, 'invalid dimension')
   
   local batchsize = input:size(1)
   local seqlen = input:size(2)

   self.gradInput:resizeAs(input):zero()

   for b = 1, batchsize do
      for s = 1, seqlen do
         local kv, ki = input[b][s]:topk(self.k, true)
         for i = 1, ki:size(1) do
            self.gradInput[{b,s,ki[i]}] = gradOutput[{b,s,ki[i]}]
         end
      end
   end

   return self.gradInput
end
