require 'hdf5'
require 'paths'
require 'rnn'
require 'nngraph'
local dl = require 'dataload'
local csv = require 'csv'

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on LAMBADA dataset')
cmd:text('Example:')
cmd:text("th train.lua --progress --earlystop 50 --cuda --device 2 --seqlen 20 --hiddensize '{200,200}' --batchsize 20 --startlr 1 --uniform 0.1 --cutoff 5 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'")
cmd:text("th examples/train.lua --cuda --cutoff 10 --batchsize 128 --seqlen 100 --hiddensize '{250,250}' --progress --device 2")
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.05, 'learning rate at t=0')
cmd:option('--minlr', 0.00001, 'minimum learning rate')
cmd:option('--saturate', 400, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--profile', false, 'profile updateOutput,updateGradInput and accGradParameters in Sequential')
cmd:option('--maxepoch', 100, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--continue', '', 'path to model for which training should be continued. Note that current options (except for device, cuda) will be ignored.')
-- rnn layer 
cmd:option('--seqlen', 50, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--inputsize', -1, 'size of lookup table embeddings. -1 defaults to hiddensize[1]')
cmd:option('--hiddensize', '{256,256}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--projsize', -1, 'size of the projection layer (number of hidden cell units for LSTMP)')
cmd:option('--dropout', 0, 'ancelossy dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--datafile', 'lambada.hdf5', 'the preprocessed hdf5 data file')
cmd:option('--vocab', 'lambada.vocab', 'the preprocessed Vocabulary file containing word index and unigram frequency')
cmd:option('--testmodel', '', 'the saved model to test')
cmd:option('--batchsize', 128, 'number of examples per batch')
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'rnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')
cmd:option('--dontsave', false, 'dont save the model')
-- unit test
cmd:option('--unittest', false, 'enable unit tests')

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
opt.inputsize = opt.inputsize == -1 and opt.hiddensize[1] or opt.inputsize
opt.id = opt.id == '' and ('lambada' .. ':' .. dl.uniqueid()) or opt.id
opt.version = 6 -- better NCE bias initialization + new default hyper-params
if not opt.silent then
  table.print(opt)
end

if opt.cuda then -- do this before building model to prevent segfault
  require 'cunn' 
  cutorch.setDevice(opt.device)
end 

local xplog, lm, criterion
if opt.continue ~= '' then
  xplog = torch.load(opt.continue)
  xplog.opt.cuda = opt.cuda
  xplog.opt.device = opt.device
  opt = xplog.opt
  lm = xplog.model.module
  -- prevent re-casting bug
  for i,lookup in ipairs(lm:findModules('nn.LookupTableMaskZero')) do
    lookup.__input = nil
  end
  criterion = xplog.criterion
  assert(opt)
end

--[[ data set ]]--

function compare_tensor_length(left, right)
  return left[1]:size(1) > right[1]:size(1) -- descending
end

function load_vocab(vocab_file)
  ivocab = {}
  vocab = {}
  wordfreq = {}
  local vocabFile = csv.open(vocab_file)
  for fields in vocabFile:lines() do
    local idx = tonumber(fields[1]) -- csv uses string by default
    if idx ~= 0 then -- 0 is only used to separate training examples
      local word = fields[2]
      local wordFreq = tonumber(fields[3])
      ivocab[idx] = word
      vocab[word] = idx
      wordfreq[#wordfreq + 1] = wordFreq  
    end
  end
  wordfreq = torch.LongTensor(wordfreq)
  return ivocab, vocab, wordfreq
end

-- load data into a table of tensors sorted by length, then group
-- each consecutive chunk into batches. pad_zeros = true will pad
-- zeros to the last batch if needed, otherwise the last batch is
-- [end - batchsize, end]
function loadData(tensor_data, pad_zeros)
  local tensor_table = {}
  local tensor_current = {}
  local tensor_length = tensor_data:size(1)

  for i = 1,tensor_length-1 do
    if tensor_data[i + 1] == 0 then
      tensor_table[#tensor_table + 1] = {torch.LongTensor(tensor_current), tensor_data[i]}
      tensor_current = {}
    else
      tensor_current[#tensor_current + 1] = tensor_data[i]
    end
  end
  if #tensor_current > 0 then -- handle residual
    tensor_table[#tensor_table + 1] = {torch.LongTensor(tensor_current), tensor_data[tensor_length]}
  end

  table.sort(tensor_table, compare_tensor_length)

  local data = {}
  local targets = {}
  local b = 1
  while b < #tensor_table do
    local bstart
    local bend
    if pad_zeros then
      bstart = b
      bend = b + opt.batchsize - 1
    else
      bend = math.min(b + opt.batchsize - 1, #tensor_table)
      bstart = bend - opt.batchsize + 1
    end
    local max_length = tensor_table[bstart][1]:size(1)
    local btensor = torch.LongTensor(max_length, opt.batchsize):zero()
    local btargets = torch.LongTensor(opt.batchsize):zero()
    for bi = 1,opt.batchsize do
      local i = bstart + bi - 1
      if i <= #tensor_table then
        local cur = tensor_table[i][1]
        btensor[{{max_length - cur:size(1) + 1, max_length}, bi}] = cur
        btargets[bi] = tensor_table[i][2]
      end
    end
    b = bend
    data[#data + 1] = btensor
    targets[#targets + 1] = btargets
  end
  return data, targets
end

function test_model(model_file)
  -- load model for computing accuracy & perplexity for target answers
  local metadata = torch.load(model_file)
  local target_mod = metadata.targetmodule
  local batch_size = metadata.opt.batchsize
  local model = metadata.model

  model:forget()
  model:evaluate()

  local logsoftmax = nn.LogSoftMax()
  if opt.cuda then
    logsoftmax:cuda()
  end
  local sumErr = 0
  local correct = 0
  local num_examples = 0
  for i = 1,#testset do
    local inputs = testset[i]
    local target = test_targets[i]
    local outputs = model:forward(inputs)
    local scores = logsoftmax:forward(outputs)

    for b = 1,outputs:size(1) do
      if target[b] ~= 0 then
        local logprob, pred_index = torch.max(scores[b], 1)
        if pred_index == target[b] then
          correct = correct + 1
        end
        sumErr = sumErr + scores[b][target[b]]
        num_examples = num_examples + 1
      end
    end

    if opt.progress then
      xlua.progress(i, #testset)
    end
  end

  local accuracy = correct / num_examples
  local perplexity = torch.exp(-sumErr / num_examples)
  print('Test Accuracy = '..accuracy..' ('..correct..' out of '..num_examples..'), Perplexity = '..perplexity)

  local test_result = {}
  test_result.accuracy = accuracy
  test_result.perplexity = perplexity

  local test_result_file = model_file..'.testresult'
  print('Saving test result file to '..test_result_file)
  torch.save(test_result_file, test_result)
end

data = hdf5.open(opt.datafile, 'r'):all()

if #opt.testmodel > 0 then
  testset, test_targets = loadData(data.test, true)
  test_model(opt.testmodel)
  os.exit()
end

ivocab, vocab, word_freq = load_vocab('lambada.vocab')

trainset,   train_targets   = loadData(data.train)
validset,   valid_targets   = loadData(data.valid)
controlset, control_targets = loadData(data.control)
testset,    test_targets    = loadData(data.test, true)

-- print('testset')
-- print(testset:size())
-- print('answerset')
-- print(answerset:view(1,-1))
-- print('testset:sub(1,200)')
-- print(testset:sub(1,100))
-- print('trainset:size()')
-- print(trainset:size())
-- print(trainset:sub(100,150):size())
-- print('validset:size()')
-- print(validset:size())
-- print(validset:sub(100,150):size())
-- print('testset:size()')
-- print(testset:size())
-- print(testset:sub(100,150):size())

if not opt.silent then 
  print("Vocabulary size : "..#ivocab) 
  print("Using batch size of "..opt.batchsize)
end

--[[ language model ]]--

if not lm then
  lm = nn.Sequential()

  -- input layer (i.e. word embedding space)
  local lookup = nn.LookupTableMaskZero(#ivocab, opt.inputsize)
  lookup.maxnormout = -1 -- prevent weird maxnormout behaviour
  lm:add(lookup) -- input is seqlen x batchsize
  if opt.dropout > 0 then
      lm:add(nn.Dropout(opt.dropout))
  end

  -- rnn layers
  local inputsize = opt.inputsize
  for i,hiddensize in ipairs(opt.hiddensize) do
    -- this is a faster version of nn.Sequencer(nn.FastLSTM(inpusize, hiddensize))
    local rnn =  opt.projsize < 1 and nn.SeqLSTM(inputsize, hiddensize) 
      or nn.SeqLSTMP(inputsize, opt.projsize, hiddensize) -- LSTM with a projection layer
    rnn.maskzero = true
    lm:add(rnn)
    if opt.dropout > 0 then
      lm:add(nn.Dropout(opt.dropout))
    end
    inputsize = hiddensize
  end

  print('#ivocab')
  print(#ivocab)

  lm:add(nn.SplitTable(1)):add(nn.SelectTable(-1)):add(nn.Linear(inputsize, #ivocab))

  -- output layer
  -- print('unigram:size()')
  -- print(unigram:size())
  -- print('inputsize')
  -- print(inputsize)
  -- print('#trainset.ivocab')
  -- print(#trainset.ivocab)

  -- don't remember previous state between batches since
  -- every example is entirely contained in a batch
  lm:remember('neither')

  if opt.uniform > 0 then
    for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
    end
  end
end

if opt.profile then
   lm:profile()
end

if not opt.silent then
   print"Language Model:"
   print(lm)
end

if not (criterion) then
  criterion = nn.CrossEntropyCriterion()
end

--[[ CUDA ]]--

if opt.cuda then
  lm:cuda()
  criterion:cuda()
end

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
if not xplog then
  xplog = {}
  xplog.opt = opt -- save all hyper-parameters and such
  xplog.dataset = 'Lambada'
  xplog.vocab = trainset.vocab
  -- will only serialize params
  xplog.model = nn.Serial(lm)
  xplog.model:mediumSerial()
  xplog.criterion = criterion
  -- keep a log of NLL for each epoch
  xplog.trainloss = {}
  xplog.valloss = {}
  -- will be used for early-stopping
  xplog.minvalloss = 99999999
  xplog.epoch = 0
  paths.mkdir(opt.savepath)
end
local ntrial = 0

local epoch = xplog.epoch+1
opt.lr = opt.lr or opt.startlr
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print("")
  print("Epoch #"..epoch.." :")

  -- 1. training
   
  local a = torch.Timer()
  lm:training()
  local sumErr = 0

  local rand_batches = torch.randperm(#trainset)
  local nbatches = rand_batches:size(1)
  for ri = 1,nbatches do
    local i = rand_batches[ri]
    inputs = trainset[i]
    targets = train_targets[i]
    -- forward
    local outputs = lm:forward(inputs)
    local err = criterion:forward(outputs, targets)
    sumErr = sumErr + err
    -- backward 
    local gradOutputs = criterion:backward(outputs, targets)
    local a = torch.Timer()
    lm:zeroGradParameters()
    lm:backward(inputs, gradOutputs)
    
    -- update
    if opt.cutoff > 0 then
      local norm = lm:gradParamClip(opt.cutoff) -- affects gradParams
      opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
    end
    lm:updateGradParameters(opt.momentum) -- affects gradParams
    lm:updateParameters(opt.lr) -- affects params
    lm:maxParamNorm(opt.maxnormout) -- affects params

    if opt.progress then
      xlua.progress(ri, nbatches)
    end

    if ri % 2000 == 0 then
      collectgarbage()
    end
  end
   
   -- learning rate decay
  if opt.schedule then
    opt.lr = opt.schedule[epoch] or opt.lr
  else
    opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
  end
  opt.lr = math.max(opt.minlr, opt.lr)
   
  if not opt.silent then
    print("learning rate", opt.lr)
    if opt.meanNorm then
      print("mean gradParam norm", opt.meanNorm)
    end
  end

  if cutorch then cutorch.synchronize() end
  local speed = nbatches*opt.batchsize/a:time().real
  print(string.format("Speed : %f words/second; %f ms/word", speed, 1000/speed))

  local loss = sumErr/nbatches
  print("Training error : "..loss)

  xplog.trainloss[epoch] = loss

  -- 2. cross-validation

  lm:evaluate()
  local sumErr = 0

  for i = 1, #validset do
    local inputs = validset[i]
    local targets = valid_targets[i]
    local outputs = lm:forward(inputs)
    local err = criterion:forward(outputs, targets)
    sumErr = sumErr + err
     
    if opt.progress then
      xlua.progress(i, #validset)
    end
  end

  local validloss = sumErr/#validset
  print("Validation error : "..validloss)

  xplog.valloss[epoch] = validloss
  ntrial = ntrial + 1

  -- early-stopping
  if validloss < xplog.minvalloss then
    -- save best version of model
    xplog.minvalloss = validloss
    xplog.epoch = epoch 
    local filename = paths.concat(opt.savepath, opt.id..'.t7')
    if not opt.dontsave then
      print("Found new minima. Saving to "..filename)
      torch.save(filename, xplog)
    end
    ntrial = 0
  elseif ntrial >= opt.earlystop then
    print("No new minima found after "..ntrial.." epochs.")
    print("Stopping experiment.")
    print("Best model can be found in "..paths.concat(opt.savepath, opt.id..'.t7'))
    os.exit()
  end

  collectgarbage()
  epoch = epoch + 1
end

test_model(paths.concat(opt.savepath, opt.id..'.t7'))