require 'nn'

-- zero out diagonal values
local MakeValuesZero, parent = torch.class('nn.MakeValuesZero', 'nn.Module')

function MakeValuesZero:__init(values1, values2)
   parent.__init(self)
   self.values1 = values1
   self.values2 = values2
   self._output = { torch.LongTensor(), torch.LongTensor() }
   self.gradInput = { torch.Tensor(), torch.Tensor() }
end

function MakeValuesZero:updateOutput(input)
   -- input is table of 2 elements of size seqlen x batchsize
   self._output[1]:resizeAs(input[1]):copy(input[1])
   self._output[2]:resizeAs(input[2]):copy(input[2])
   for b = 1, input[2]:size(1) do
      for s = 1, input[2]:size(2) do
         if self.values1[input[1][b][s]] == nil and self.values2[input[2][b][s]] == nil then
            self._output[1][b][s] = 0
            self._output[2][b][s] = 0
         end
      end
   end
   self.output = self._output
   return self.output
end

function MakeValuesZero:updateGradInput(input, gradOutput)
   -- input size seqlen x batchsize
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput[1])
   self.gradInput[2]:resizeAs(input[2]):copy(gradOutput[2])

   for b = 1, input[2]:size(1) do
      for s = 1, input[2]:size(2) do
         if self.values1[input[1][b][s]] == nil and self.values2[input[2][b][s]] == nil then
            self.gradInput[1][b][s] = 0
            self.gradInput[2][b][s] = 0
         end
      end
   end
   return self.gradInput
end
