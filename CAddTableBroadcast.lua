require 'nn'

local CAddTableBroadcast, parent = torch.class('nn.CAddTableBroadcast', 'nn.Module')

function CAddTableBroadcast:__init(ip)
   parent.__init(self)
   self.inplace = ip
   self.gradInput = {}
end

function CAddTableBroadcast:updateOutput(input)
   if self.inplace then
      self.output:set(input[1])
   else
      self.output:resizeAs(input[1]):copy(input[1])
   end
   local dim1 = input[1]:size(1)
   local dim2 = input[1]:size(2)
   local dim3 = input[1]:size(3)
   for i=2,#input do
      self.output:add(input[i]:repeatTensor(dim1,1):reshape(dim1,dim2,dim3))
   end
   return self.output
end

function CAddTableBroadcast:updateGradInput(input, gradOutput)
   self.gradInput[1] = self.gradInput[1] or input[1].new()
   if self.inplace then
      self.gradInput[1]:set(gradOutput)
   else
      self.gradInput[1]:resizeAs(input[1]):copy(gradOutput)
   end

   for i=2,#input do
      self.gradInput[i] = self.gradInput[i] or input[i].new()
      if self.inplace then
         self.gradInput[i]:set(gradOutput:sum(1):squeeze())
      else
         self.gradInput[i]:resizeAs(input[i]):copy(gradOutput:sum(1):squeeze())
      end
   end

   for i=#input+1, #self.gradInput do
      self.gradInput[i] = nil
   end

   return self.gradInput
end