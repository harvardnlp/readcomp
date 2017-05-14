require 'nn'

local ZeroToNegInf, parent = torch.class('nn.ZeroToNegInf', 'nn.Module')

function ZeroToNegInf:__init(ip)
   parent.__init(self)
   self.inplace = ip
end

function ZeroToNegInf:updateOutput(input)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   self.output[input:eq(0)] = torch.log(0)
   return self.output
end

function ZeroToNegInf:updateGradInput(input, gradOutput)
   self.gradInput = self.gradInput or input.new()
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(input):copy(gradOutput)
   end

   self.gradInput[input:eq(0)] = 0

   return self.gradInput
end