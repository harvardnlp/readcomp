require 'nn'

-- extract non-zero final states of BRNN when inputs contain zeros
-- output = concat( final non-zero state of forward(x), final non-zero state of backward(x) )
local SeqBRNNAtLoc, parent = torch.class('nn.SeqBRNNAtLoc', 'nn.Module')

function SeqBRNNAtLoc:__init(qidx)
   parent.__init(self)
   self.gradInput = torch.Tensor()
   self.qidx = qidx -- index to look for
end


function SeqBRNNAtLoc:setStuff(seqinp, tens_loc, bidx)
  self.seqinp = seqinp
  self.tens_loc = tens_loc
  self.batch_idx = bidx
end

function SeqBRNNAtLoc:getAnsLoc(b)
  local cur_offset = self.tens_loc[self.batch_idx+b-1][1]
  local cur_context_length = self.tens_loc[self.batch_idx+b-1][2]
  local ans_loc = cur_offset + cur_context_length
  return ans_loc
end

function SeqBRNNAtLoc:updateOutput(input)
   -- expects input of size batchsize x seqlen x (2 * hiddensize)
   -- where the first hiddensize chunk is from forward RNN pass, and the second from backward pass
   local batchsize = input:size(1)
   local seqlen = input:size(2)
   local hiddensize2 = input:size(3)
   local hiddensize = hiddensize2 / 2

   self.output:resize(batchsize, 2 * hiddensize):zero()

   for b = 1, batchsize do
      local ans_loc = self:getAnsLoc(b)
      beforeLocIdx, afterLocIdx = self:getBeforeAfterRows(seqlen, ans_loc)
      --firstNonzeroExample, lastNonzeroExample = self:getFirstLastNonZeroRow(input[b], seqlen, hiddensize)
      -- before is always kosher
      self.output[{b, {1, hiddensize}}] = input[{b, beforeLocIdx, {1, hiddensize}}]
      if afterLocIdx <= seqlen then
        self.output[{b, {hiddensize + 1, hiddensize2}}] = input[{b, afterLocIdx, {hiddensize + 1, hiddensize2}}]
      end
      -- if firstNonzeroExample > 0 and lastNonzeroExample > 0 then
      --    self.output[{b, {1, hiddensize}}] = input[{b, lastNonzeroExample, {1, hiddensize}}]
      --    self.output[{b, {hiddensize + 1, hiddensize2}}] = input[{b, firstNonzeroExample, {hiddensize + 1, hiddensize2}}]
      -- end
   end

   return self.output
end

function SeqBRNNAtLoc:updateGradInput(input, gradOutput)
   local batchsize = input:size(1)
   local seqlen = input:size(2)
   local hiddensize2 = input:size(3)
   local hiddensize = hiddensize2 / 2

   self.gradInput:resizeAs(input):zero()

   for b = 1, batchsize do
     local ans_loc = self:getAnsLoc(b)
     beforeLocIdx, afterLocIdx = self:getBeforeAfterRows(seqlen, ans_loc)
     self.gradInput[{b, beforeLocIdx, {1, hiddensize}}] = gradOutput[{b, {1, hiddensize}}]
     if afterLocIdx <= seqlen then
       self.gradInput[{b, afterLocIdx, {hiddensize + 1, hiddensize2}}] = gradOutput[{b, {hiddensize + 1, hiddensize2}}]
     end
      -- firstNonzeroExample, lastNonzeroExample = self:getFirstLastNonZeroRow(input[b], seqlen, hiddensize)
      -- if firstNonzeroExample > 0 and lastNonzeroExample > 0 then
      --    self.gradInput[{b, lastNonzeroExample, {1, hiddensize}}] = gradOutput[{b, {1, hiddensize}}]
      --    self.gradInput[{b, firstNonzeroExample, {hiddensize + 1, hiddensize2}}] = gradOutput[{b, {hiddensize + 1, hiddensize2}}]
      -- end
   end

   return self.gradInput
end

function SeqBRNNAtLoc:getBeforeAfterRows(seqlen, answer_loc)
   -- seqlen is the number of RNN states
   local beforeLocIdx = 0
   local afterLocIdx = 0

   for s = 1, seqlen do -- start at answer
     if self.seqinp[answer_loc-s] == self.qidx then
       beforeLocIdx = seqlen - s
       afterLocIdx = seqlen - s + 2
       break
     end
   end

   return beforeLocIdx, afterLocIdx
end
