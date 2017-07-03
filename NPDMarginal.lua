require 'nn'

local NPDMarginal, parent = torch.class('nn.NPDMarginal', 'nn.Module')

function NPDMarginal:__init()
   parent.__init(self)
   self.gradInput = {}
end

function NPDMarginal:updateOutput(input)
   -- input is batch x ̣̣̣̣̣̣̣̣(seqlen + 1) x seqlen
   -- first row contains scores for root nodes
   self.output:resizeAs(input[1]):copy(input[1])

   local L = -torch.exp(input)
   L[{{},1}] = -L[{{},1}]
   local diag = torch.sum(L[{{},{2,-1}}],2)
   
   for b = 1, input:size(1) do
      for i = 1, input:size(3) do
         L[b][i+1][i] = diag[b][1][i]
      end
   end

   local dim1 = input[1]:size(1)
   local dim2 = input[1]:size(2)
   local dim3 = input[1]:size(3)
   for i=2,#input do
      self.output:add(input[i]:repeatTensor(dim1,1):reshape(dim1,dim2,dim3))
   end
   return self.output
end

function NPDMarginal:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)

   for i=2,#input do
      self.gradInput[i] = self.gradInput[i] or input[i].new()
      self.gradInput[i]:resizeAs(input[i]):copy(gradOutput:sum(1):squeeze())
   end

   for i=#input+1, #self.gradInput do
      self.gradInput[i] = nil
   end

   return self.gradInput
end