require 'nn'

local MaxNodeMarginal, parent = torch.class('nn.MaxNodeMarginal', 'nn.Module')

function MaxNodeMarginal:__init(max_state)
   parent.__init(self)
   self.max_state = max_state or 1
end

function MaxNodeMarginal:updateOutput(input)
   -- input is batch x seqlen x nstates
   -- output is batch x seqlen

   local batch = input:size(1)
   local seqlen = input:size(2)
   local nstates = input:size(3)

   -- print('self.max_state')
   -- print(self.max_state)

   self.output:resize(batch, seqlen):zero()
   for b = 1, batch do
      for i = 1, seqlen do
         if input[b][i][self.max_state] == input[b][i]:max() then
            self.output[b][i] = input[b][i][self.max_state]
         end
      end
   end
   -- local x = nn.Normalize(1):forward(self.output)
   -- print('x')
   -- print(x)
   -- print('self.output')
   -- print(self.output)
   return self.output
end

function MaxNodeMarginal:updateGradInput(input, gradOutput)
   local batch = input:size(1)
   local seqlen = input:size(2)
   local nstates = input:size(3)

   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):zero()

   for b = 1, batch do
      for i = 1, seqlen do
         if input[b][i][self.max_state] == input[b][i]:max() then
            self.gradInput[b][i][self.max_state] = gradOutput[b][i]
         end
      end
   end

   return self.gradInput
end