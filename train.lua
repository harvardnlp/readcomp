require 'hdf5'
require 'paths'
require 'rnn'
require 'nngraph'
local dl = require 'dataload'
local csv = require 'csv'
assert(nn.NCEModule and nn.NCEModule.version and nn.NCEModule.version >= 6, "update dpnn : luarocks install dpnn")

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on LAMBADA dataset')
cmd:text('Example:')
cmd:text("th train.lua --progress --earlystop 50 --cuda --device 2 --seqlen 20 --hiddensize '{200,200}' --batchsize 20 --startlr 1 --uniform 0.1 --cutoff 5 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'")
cmd:text("th examples/train.lua --cuda --trainsize 400000 --validsize 40000 --cutoff 10 --batchsize 128 --seqlen 100 --hiddensize '{250,250}' --progress --device 2")
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
cmd:option('--maxepoch', 1000, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--k', 100, 'how many noise samples to use for NCE')
cmd:option('--continue', '', 'path to model for which training should be continued. Note that current options (except for device, cuda and tiny) will be ignored.')
cmd:option('--Z', 1, 'normalization constant for NCE module (-1 approximates it from first batch).')
cmd:option('--rownoise', false, 'sample k noise samples for each row for NCE module')
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
cmd:option('--trainsize', 400000, 'number of train time-steps seen between each epoch')
cmd:option('--validsize', 40000, 'number of valid time-steps used for early stopping and cross-validation') 
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'rnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')
cmd:option('--tiny', false, 'use train_tiny.th7 training file')
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

local xplog, lm, criterion, targetmodule
if opt.continue ~= '' then
  xplog = torch.load(opt.continue)
  xplog.opt.cuda = opt.cuda
  xplog.opt.device = opt.device
  xplog.opt.tiny = opt.tiny
  opt = xplog.opt
  lm = xplog.model.module
  -- prevent re-casting bug
  for i,lookup in ipairs(lm:findModules('nn.LookupTableMaskZero')) do
    lookup.__input = nil
  end
  criterion = xplog.criterion
  targetmodule = xplog.targetmodule
  assert(opt)
end

--[[ data set ]]--

function loadData(tensor_data, data_type)
  local tensor_table = {}
  local tensor_current = {}
  local tensor_length = tensor_data:size(1)
  for i = 1,tensor_length do
    if tensor_data[i] == 0 then
      tensor_table[#tensor_table + 1] = torch.LongTensor(tensor_current)
      tensor_current = {}
    else
      tensor_current[#tensor_current + 1] = tensor_data[i]
    end
  end
  if #tensor_current > 0 then -- handle residual
    tensor_table[#tensor_table + 1] = torch.LongTensor(tensor_current)
  end
  if data_type == 'test' then
  	local max_length = 0
  	for i = 1, #tensor_table do
  		max_length = math.max(max_length, tensor_table[i]:size(1) - 1)
  	end
  	local test_tensor = torch.LongTensor(max_length, #tensor_table)
    local answer_tensor = torch.LongTensor(#tensor_table)
  	for i = 1, #tensor_table do
  		local t = tensor_table[i]
  		local len = t:size(1)
  		for l = 1, len - 1 do
  			test_tensor[l][i] = t[l]
  		end
      answer_tensor[i] = t[len]
  	end
  	return test_tensor, answer_tensor
  end
  local d = dl.MultiSequence(tensor_table, opt.batchsize)
  if data_type == 'train' then -- load ivocab, vocab and unigram word frequency
    d.ivocab = {}
    d.vocab = {}
    d.wordfreq = {}
    local vocabFile = csv.open('lambada.vocab')
    for fields in vocabFile:lines() do
      local idx = tonumber(fields[1]) -- csv uses string by default
      if idx ~= 0 then -- 0 is only used to separate training examples
        local word = fields[2]
        local wordFreq = tonumber(fields[3])
        d.ivocab[idx] = word
        d.vocab[word] = idx
        d.wordfreq[#d.wordfreq + 1] = wordFreq  
      end
    end
    d.wordfreq = torch.LongTensor(d.wordfreq)
  end
  return d
end

function test_model(model_file)
  -- load model for computing accuracy & perplexity for target answers
  local metadata = torch.load(model_file)
  local target_mod = metadata.targetmodule
  local batch_size = metadata.opt.batchsize
  local model = metadata.model

  model:forget()
  model:evaluate()
  model:findModules('nn.NCEModule')[1].normalized = true
  model:findModules('nn.NCEModule')[1].logsoftmax = true

  local sumErr = 0
  local correct = 0
  local max_length = testset:size(1)
  local num_examples = testset:size(2)
  local num_batches = math.ceil(num_examples / batch_size)
  local rounded_num_examples = num_batches * batch_size
  local num_pad_examples = rounded_num_examples - num_examples

  local dummy_targets = target_mod:forward(torch.LongTensor(max_length, batch_size))

  if num_pad_examples > 0 then
    testset = torch.cat(testset, torch.zeros(max_length, num_pad_examples):long())
  end

  for i = 1,num_batches do
    local batch_start = (i - 1) * batch_size + 1
    local batch_end = i * batch_size
    local batch_test = testset:sub(1, max_length, batch_start, batch_end)
    -- print(batch_test)
    local outputs = model:forward({batch_test, dummy_targets})

    -- get predictions for the last word in the sentence
    for b = 1,batch_size do
      local example_index = batch_start + b - 1
      if example_index > num_examples then
        break
      end
      local last_index = 0
      for l = 1,max_length do
        if testset[l][example_index] == 0 then
          last_index = l - 1
          break
        end
      end
      if last_index > 0 then
        -- print('last index = '..last_index..', size of score array = '..outputs[last_index][b]:size(1)..'')
        local scores = outputs[last_index][b]
        if opt.unittest then
          assert(math.abs(torch.sum(torch.exp(scores)) - 1) < 1e-6, 'Invalid logprob scores: '..torch.sum(torch.exp(scores)))
        end
        local logprob,pred_index = torch.max(scores, 1)
        local correct_index = answerset[example_index]
        if pred_index == correct_index then
          correct = correct + 1
        end
        local correct_logprob = outputs[last_index][b][correct_index]
        sumErr = sumErr + correct_logprob
      end
    end

    if opt.progress then
      xlua.progress(i, num_batches)
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
  testset, answerset = loadData(data.test, 'test')
  test_model(opt.testmodel)
  os.exit()
end

trainset   = loadData(data.train, 'train')
validset   = loadData(data.valid)
controlset = loadData(data.control)

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
  print("Vocabulary size : "..#trainset.ivocab) 
  print("Train set split into "..opt.batchsize.." sequences of length "..trainset:size())
end

--[[ language model ]]--

if not lm then
  lm = nn.Sequential()

  -- input layer (i.e. word embedding space)
  local lookup = nn.LookupTableMaskZero(#trainset.ivocab, opt.inputsize)
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

  lm:add(nn.SplitTable(1))

  -- output layer
  local unigram = trainset.wordfreq:float()
  -- print('unigram:size()')
  -- print(unigram:size())
  -- print('inputsize')
  -- print(inputsize)
  -- print('#trainset.ivocab')
  -- print(#trainset.ivocab)
  -- print('opt.k')
  -- print(opt.k)
  -- print('opt.Z')
  -- print(opt.Z)
  local ncemodule = nn.NCEModule(inputsize, #trainset.ivocab, opt.k, unigram, opt.Z)
  ncemodule.batchnoise = not opt.rownoise

  -- NCE requires {input, target} as inputs
  lm = nn.Sequential()
    :add(nn.ParallelTable()
       :add(lm):add(nn.Identity()))
    :add(nn.ZipTable()) -- {{x1,x2,...}, {t1,t2,...}} -> {{x1,t1},{x2,t2},...}

  -- encapsulate stepmodule into a Sequencer
  lm:add(nn.Sequencer(nn.MaskZero(ncemodule, 1)))
   
  -- remember previous state between batches
  lm:remember()

  if opt.uniform > 0 then
    for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
    end
    ncemodule:reset()
  end
end

if opt.profile then
   lm:profile()
end

if not opt.silent then
   print"Language Model:"
   print(lm)
end

if not (criterion and targetmodule) then
  --[[ loss function ]]--

  local crit = nn.MaskZeroCriterion(nn.NCECriterion(), 0)

   -- target is also seqlen x batchsize.
  targetmodule = nn.SplitTable(1)
  if opt.cuda then
    targetmodule = nn.Sequential()
      :add(nn.Convert())
      :add(targetmodule)
  end
    
  criterion = nn.SequencerCriterion(crit)
end

--[[ CUDA ]]--

if opt.cuda then
  lm:cuda()
  criterion:cuda()
  targetmodule:cuda()
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
  xplog.targetmodule = targetmodule
  -- keep a log of NLL for each epoch
  xplog.trainnceloss = {}
  xplog.valnceloss = {}
  -- will be used for early-stopping
  xplog.minvalnceloss = 99999999
  xplog.epoch = 0
  paths.mkdir(opt.savepath)
end
local ntrial = 0

local epoch = xplog.epoch+1
opt.lr = opt.lr or opt.startlr
opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize
while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print("")
  print("Epoch #"..epoch.." :")

  -- 1. training
   
  local a = torch.Timer()
  lm:training()
  local sumErr = 0
  for i, inputs, targets in trainset:subiter(opt.seqlen, opt.trainsize) do
    targets = targetmodule:forward(targets)
    inputs = {inputs, targets}
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
      xlua.progress(i, opt.trainsize)
    end

    if i % 2000 == 0 then
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
  local speed = opt.trainsize*opt.batchsize/a:time().real
  print(string.format("Speed : %f words/second; %f ms/word", speed, 1000/speed))

  local nceloss = sumErr/opt.trainsize
  print("Training error : "..nceloss)

  xplog.trainnceloss[epoch] = nceloss

  -- 2. cross-validation

  lm:evaluate()
  local sumErr = 0
  for i, inputs, targets in validset:subiter(opt.seqlen, opt.validsize) do
    targets = targetmodule:forward(targets)
    local outputs = lm:forward{inputs, targets}
    local err = criterion:forward(outputs, targets)
    sumErr = sumErr + err
     
    if opt.progress then
      xlua.progress(i, opt.validsize)
    end
  end

  local nceloss = sumErr/opt.validsize
  print("Validation error : "..nceloss)

  xplog.valnceloss[epoch] = nceloss
  ntrial = ntrial + 1

  -- early-stopping
  if nceloss < xplog.minvalnceloss then
    -- save best version of model
    xplog.minvalnceloss = nceloss
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