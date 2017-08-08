require 'nn'

-- Sinusoid Positional Encoding from https://arxiv.org/abs/1706.03762
local SinusoidPositionEncoding, parent = torch.class('nn.SinusoidPositionEncoding', 'nn.Module')

function SinusoidPositionEncoding:__init(maxLen, hiddenSize)
   parent.__init(self)
   self.gradInput = torch.Tensor()

   self.pe = torch.zeros(maxLen, hiddenSize)
   for pos = 1, maxLen do
      for i = 1, hiddenSize do
         if i % 2 == 0 then
            self.pe[pos][i] = math.sin(pos / math.pow(10000, 2 * i / hiddenSize))
         else
            self.pe[pos][i] = math.cos(pos / math.pow(10000, 2 * i / hiddenSize))
         end
      end
   end
end

function SinusoidPositionEncoding:updateOutput(input)
   -- input size seqlen x batchsize x hiddensize
   assert(input:dim() == 3, 'invalid dimension')

   local seqlen = input:size(1)
   local batchsize = input:size(2)
   local hidsize = input:size(3)

   self.output:resizeAs(input):copy(input)
   self.output:add(self.pe[{{1,seqlen}}]:view(seqlen, 1, hidsize):expand(seqlen, batchsize, hidsize))

   return self.output
end

function SinusoidPositionEncoding:updateGradInput(input, gradOutput)
   -- input size seqlen x batchsize x hiddensize
   assert(input:dim() == 3, 'invalid dimension')

   self.gradInput:resizeAs(input):copy(gradOutput)
   return self.gradInput
end
