require 'nn'

-- extract non-zero final states of BRNN when inputs contain zeros
-- output = concat( final non-zero state of forward(x), final non-zero state of backward(x) )
local MaskZeroSeqBRNNFinal, parent = torch.class('nn.MaskZeroSeqBRNNFinal', 'nn.Module')

function MaskZeroSeqBRNNFinal:__init()
   parent.__init(self)
   self.gradInput = torch.Tensor()
end

function MaskZeroSeqBRNNFinal:updateOutput(input)
   -- expects input of size batchsize x seqlen x (2 * hiddensize)
   -- where the first hiddensize chunk is from forward RNN pass, and the second from backward pass
   local batchsize = input:size(1)
   local seqlen = input:size(2)
   local hiddensize2 = input:size(3)
   local hiddensize = hiddensize2 / 2

   self.output:resize(batchsize, 2 * hiddensize):zero()

   for b = 1, batchsize do
      firstNonzeroExample, lastNonzeroExample = self:getFirstLastNonZeroRow(input[b], seqlen, hiddensize)
      if firstNonzeroExample > 0 and lastNonzeroExample > 0 then
         self.output[{b, {1, hiddensize}}] = input[{b, lastNonzeroExample, {1, hiddensize}}]
         self.output[{b, {hiddensize + 1, hiddensize2}}] = input[{b, firstNonzeroExample, {hiddensize + 1, hiddensize2}}]
      end
   end

   return self.output
end

function MaskZeroSeqBRNNFinal:updateGradInput(input, gradOutput)
   local batchsize = input:size(1)
   local seqlen = input:size(2)
   local hiddensize2 = input:size(3)
   local hiddensize = hiddensize2 / 2

   self.gradInput:resizeAs(input):zero()

   for b = 1, batchsize do
      firstNonzeroExample, lastNonzeroExample = self:getFirstLastNonZeroRow(input[b], seqlen, hiddensize)
      if firstNonzeroExample > 0 and lastNonzeroExample > 0 then
         self.gradInput[{b, lastNonzeroExample, {1, hiddensize}}] = gradOutput[{b, {1, hiddensize}}]
         self.gradInput[{b, firstNonzeroExample, {hiddensize + 1, hiddensize2}}] = gradOutput[{b, {hiddensize + 1, hiddensize2}}]
      end
   end

   return self.gradInput
end

function MaskZeroSeqBRNNFinal:getFirstLastNonZeroRow(input, seqlen, hiddensize)
   local firstNonzeroExample = 0
   local lastNonzeroExample = 0

   for s = 1, seqlen do
      for h = 1, 5 do --hiddensize do
         if firstNonzeroExample == 0 and input[s][h] ~= 0 then
            firstNonzeroExample = s
         end
         if lastNonzeroExample == 0 and input[seqlen - s + 1][h] ~= 0 then
            lastNonzeroExample = seqlen - s + 1
         end
         if firstNonzeroExample ~= 0 and lastNonzeroExample ~= 0 then
            return firstNonzeroExample, lastNonzeroExample
         end
      end
   end

   return firstNonzeroExample, lastNonzeroExample
end