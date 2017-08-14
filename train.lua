require 'hdf5'
require 'paths'
require 'rnn'
require 'nngraph'
require 'SeqBRNNP'
require 'MaskZeroSeqBRNNFinal'
require 'CAddTableBroadcast'
require 'SinusoidPositionEncoding'
require 'PositionWiseFFNN'
require 'MultiHeadAttention'
require 'LayerNorm'
require 'ShiftRight'
require 'optim'

require 'crf/Util.lua'
require 'crf/Markov.lua'
require "crf/CRF.lua"

tds = require 'tds'
local dl = require 'dataload'
local csv = require 'csv'

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train on LAMBADA dataset')
cmd:text('Example:')
cmd:text("th train.lua --progress --earlystop 50 --cuda --device 2 --maxseqlen 1024 --hiddensize '{200,200}' --batchsize 20 --uniform 0.1 --cutoff 5")
cmd:text('Options:')
-- training
cmd:option('--model', 'asr', 'type of models to train, acceptable values: {crf, asr ,ga}')
cmd:option('--gahop', 3, 'number of hops in gated attention model')
cmd:option('--adamconfig', '{0.9, 0.999}', 'ADAM hyperparameters beta1 and beta2')
cmd:option('--cutoff', 10, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--profile', false, 'profile updateOutput,updateGradInput and accGradParameters in Sequential')
cmd:option('--maxbatch', -1, 'maximum number of training batches per epoch')
cmd:option('--maxepoch', 10, 'maximum number of epochs to run')
cmd:option('--attstack', 6, 'number of multi-self-attention layers')
cmd:option('--atthead', 8, 'number of attention heads in multi-self-attention')
cmd:option('--dff', 2048, 'number of hidden units for the inner layer of position-wise FFNN')
cmd:option('--gcepoch', 1000, 'specify #-epoch interval to perform garbage collection')
cmd:option('--maxseqlen', 1024, 'maximum sequence length for context and target')
cmd:option('--earlystop', 5, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--lr', 0.001, 'learning rate')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
cmd:option('--continue', '', 'path to model for which training should be continued. Note that current options (except for device, cuda) will be ignored.')
-- rnn layer 
cmd:option('--inputsize', -1, 'size of lookup table embeddings. -1 defaults to hiddensize[1]')
cmd:option('--postsize', 80, 'size of pos_tag embeddings')
cmd:option('--hiddensize', '{128}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--rnntype', 'gru', 'type of rnn to use for encoding context and query, acceptable values: rnn/lstm')
cmd:option('--projsize', -1, 'size of the projection layer (number of hidden cell units for LSTMP)')
cmd:option('--dropout', 0.1, 'ancelossy dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--datafile', 'lambada.hdf5', 'the preprocessed hdf5 data file')
cmd:option('--testmodel', '', 'the saved model to test')
cmd:option('--batchsize', 64, 'number of examples per batch')
cmd:option('--savepath', 'models', 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')
cmd:option('--evalheuristics', false, 'evaluate heuristics approach e.g. random selection from context, most likely')
cmd:option('--dontsave', false, 'dont save the model')
cmd:option('--verbose', false, 'print verbose diagnostics messages')
cmd:option('--randomseed', 101, 'random seed')

-- unit test
cmd:option('--unittest', false, 'enable unit tests')

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.adamconfig = loadstring(" return "..opt.adamconfig)()
opt.inputsize = opt.inputsize == -1 and opt.hiddensize[1] or opt.inputsize
opt.id = opt.id == '' and (paths.basename(opt.datafile, paths.extname(opt.datafile)) .. '-' .. opt.model) or opt.id
opt.version = 6 -- better NCE bias initialization + new default hyper-params
if not opt.silent then
  table.print(opt)
end

torch.manualSeed(opt.randomseed)

if opt.cuda then -- do this before building model to prevent segfault
  require 'cunn' 
  cutorch.setDevice(opt.device)
end 

local xplog, lm
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
  assert(opt)
end

function collect_track_garbage()
  collectgarbage()
  local memory_message = string.format("CPU mem: %.2f Mb", collectgarbage('count') / 1000)
  if opt.cuda then
    freeMemory, totalMemory = cutorch.getMemoryUsage(opt.device)
    memory_message = memory_message .. string.format(", GPU mem: %.2f free / %.2f total (Mb)", freeMemory / 1048576, totalMemory / 1048576)
  end
  print(memory_message)
end

function test_answer_in_context(tensor_data, tensor_location, sample)

  print('Verify that answer is in context')

  local num_examples = tensor_location:size(1)
  local batches
  if sample == -1 then
    batches = torch.range(1, num_examples, opt.batchsize)
  else
    slices = torch.range(1, num_examples, opt.batchsize)
    randind = torch.randperm(slices:size(1))
    batches = torch.zeros(sample)
    for is = 1, sample do
      batches[is] = slices[randind[is]]
    end
    -- sample = math.min(sample, num_examples - opt.batchsize + 1)
    -- batches = torch.randperm(num_examples - opt.batchsize + 1):sub(1,sample)
  end
  local num_batches = batches:size(1)

  local num_answers = 0
  local num_correct_random = 0
  local num_correct_likely = 0
  for i = 1,num_batches do
    local batch_start = batches[i]
    local max_context_length = tensor_location[batch_start][2]
    local max_target_length = torch.max(tensor_location[{{batch_start, math.min(batch_start + opt.batchsize - 1, num_examples)}, 3}])
    local context = torch.LongTensor(max_context_length, opt.batchsize):zero()
    local target_forward  = torch.LongTensor(max_target_length, opt.batchsize):zero()
    local target_backward = torch.LongTensor(max_target_length, opt.batchsize):zero()
    local answer = torch.LongTensor(opt.batchsize):zero()
    local answer_ind = {}
    
    for idx = 1, opt.batchsize do
      answer_ind[idx] = {}
      local iexample = batch_start + idx - 1  
      if iexample <= num_examples then
        local cur_offset = tensor_location[iexample][1]
        local cur_context_length = tensor_location[iexample][2]
        local cur_target_length = tensor_location[iexample][3]

        local cur_context = tensor_data[{{cur_offset, cur_offset + cur_context_length - 1}}]
        local cur_target  = tensor_data[{{cur_offset + cur_context_length, cur_offset + cur_context_length + cur_target_length - 2}}]
        local cur_answer  = tensor_data[cur_offset + cur_context_length + cur_target_length - 1]
        
        context[{{max_context_length - cur_context:size(1) + 1, max_context_length}, idx}] = cur_context
        target_forward [{{max_target_length - cur_target:size(1) + 1, max_target_length}, idx}] = cur_target
        target_backward[{{max_target_length - cur_target:size(1) + 1, max_target_length}, idx}] = cur_target:index(1, torch.linspace(cur_target:size(1), 1, cur_target:size(1)):long()) -- reverse order
        answer[idx] = cur_answer

        for aid = 1, max_context_length do
          if context[aid][idx] == cur_answer then
            answer_ind[idx][#answer_ind[idx] + 1] = aid
          end
        end

        assert(#answer_ind[idx] ~= 0, 'answer must exist in context')
      end
    end
    collect_track_garbage()
  end
end

function allocate_data(max_context_length, batchsize)
  -- optimization: pre-allocate tensors and resize/reset when appropriate
  context      = context      and context     :resize(max_context_length, opt.batchsize):zero() or torch.LongTensor(max_context_length, opt.batchsize):zero()
  context_post = context_post and context_post:resize(max_context_length, opt.batchsize):zero() or torch.LongTensor(max_context_length, opt.batchsize):zero()
  context_extr = context_extr and context_extr:resize(max_context_length, opt.batchsize, extr_size):zero() or torch.LongTensor(max_context_length, opt.batchsize, extr_size):zero()

  answer = answer and answer:resize(opt.batchsize):zero() or torch.LongTensor(opt.batchsize):zero()
end

function loadData(tensor_data, tensor_post, tensor_extr, tensor_location, eval_heuristics, batch_index)
  -- pos_tags are features for both context and target
  -- extra features only apply to context (e.g. frequency of token in context)
  local num_examples = tensor_location:size(1)
  local num_answers = 0
  local num_correct_random = 0
  local num_correct_likely = 0

  local max_context_length = math.min(opt.maxseqlen, tensor_location[batch_index][2])
  allocate_data(max_context_length, opt.batchsize)

  local answer_ind = {}
  for idx = 1, opt.batchsize do
    answer_ind[idx] = {}
    local iexample = batch_index + idx - 1  
    if iexample <= num_examples then
      local cur_offset = tensor_location[iexample][1]
      local cur_context_length = tensor_location[iexample][2]
      local cur_capped_context_length = math.min(opt.maxseqlen, cur_context_length)

      local offset_end_context = cur_offset + cur_context_length
      local cur_context      = tensor_data[{{offset_end_context - cur_capped_context_length, offset_end_context - 1}}]
      local cur_context_post = tensor_post[{{offset_end_context - cur_capped_context_length, offset_end_context - 1}}]
      local cur_context_extr = tensor_extr[{{offset_end_context - cur_capped_context_length, offset_end_context - 1}}] -- cur_context_length x extr_size

      local cur_answer  = tensor_data[offset_end_context]

      local context_size = cur_context:size(1)

      context[{{max_context_length - context_size + 1, max_context_length}, idx}] = cur_context
      context_post[{{max_context_length - context_size + 1, max_context_length}, idx}] = cur_context_post
      context_extr[{{max_context_length - context_size + 1, max_context_length}, idx}] = cur_context_extr

      answer[idx] = cur_answer

      for aid = 1, max_context_length do
        if context[aid][idx] == cur_answer then
          answer_ind[idx][#answer_ind[idx] + 1] = aid
        end
      end

      if eval_heuristics then
        local random_answer = cur_context[torch.randperm(cur_context:size(1))[1]]

        local most_likely_answers = tds.hash()
        local most_likely_count = 0
        local most_likely_word = 0
        for i = 1, cur_context:size(1) do
          if puncs[cur_context[i]] == nil and stopwords[cur_context[i]] == nil then
            if most_likely_answers[cur_context[i]] == nil then
              most_likely_answers[cur_context[i]] = 1
            else
              most_likely_answers[cur_context[i]] = most_likely_answers[cur_context[i]] + 1
            end
            if most_likely_count < most_likely_answers[cur_context[i]] then
              most_likely_count = most_likely_answers[cur_context[i]]
              most_likely_word = cur_context[i]
            end
          end
        end
        num_answers = num_answers + 1

        if most_likely_word == cur_answer then
          num_correct_likely = num_correct_likely + 1
        end

        if random_answer == cur_answer then
          num_correct_random = num_correct_random + 1
        end
      end
    end
  end
  if eval_heuristics then
    print('Heuristics accuracy')
    print('num_answers')
    print(num_answers)
    print('num_correct_likely')
    print(num_correct_likely)
    print('num_correct_random')
    print(num_correct_random)
  end

  if opt.cuda then
    context = context:cuda()
    context_post = context_post:cuda()
    context_extr = context_extr:cuda()
  end

  contexts = { {context:t(), context_post:t()}, context_extr:transpose(1,2) }
  return contexts, answer, answer_ind
end

function test_model(saved_model_file, dump_name, tensor_data, tensor_post, tensor_extr, tensor_location)
  local metadata
  local batch_size = opt.batchsize
  local model = lm
  local model_file = saved_model_file and saved_model_file or paths.concat(opt.savepath, opt.id..'.t7')

  if not lm then
    -- load model for computing accuracy & perplexity for target answers
    metadata = torch.load(model_file)
    batch_size = metadata.opt.batchsize
    model = metadata.model
    puncs = metadata.puncs -- punctuations
    attention_layer = nn.SoftMax()
    if opt.cuda then
      attention_layer:cuda()
    end
  end

  model:forget()
  model:evaluate()

  dump_name       = dump_name       and dump_name       or 'test'
  tensor_data     = tensor_data     and tensor_data     or data.test_data
  tensor_post     = tensor_post     and tensor_post     or data.test_post
  tensor_extr     = tensor_extr     and tensor_extr     or data.test_extr
  tensor_location = tensor_location and tensor_location or data.test_location

  local all_batches = torch.range(1, tensor_location:size(1), opt.batchsize)
  local ntestbatches = all_batches:size(1)

  local correct = 0
  local num_examples = 0
  for i = 1, ntestbatches do
    local tests_con, tests_ans, tests_ans_ind = loadData(tensor_data, tensor_post, tensor_extr, tensor_location, false, all_batches[i])

    local inputs = tests_con
    local answer = tests_ans
    local in_words = inputs[1][1]
    local outputs = mask_attention(in_words, model:forward(inputs))

    local predictions = answer.new():resizeAs(answer):zero()
    local truth = {}
    -- compute attention sum for each word in the context, except for punctuation symbols
    for b = 1,outputs:size(1) do
      if answer[b] ~= 0 then
        local word_to_prob = tds.hash()
        local max_prob = 0
        local max_word = 0
        for iw = 1, outputs:size(2) do
          local word = in_words[b][iw]
          if word ~= 0 then
            if word_to_prob[word] == nil then
              word_to_prob[word] = 0
            end
            word_to_prob[word] = word_to_prob[word] + outputs[b][iw]
            if max_prob < word_to_prob[word] then
              max_prob = word_to_prob[word]
              max_word = word
            end
          end
        end
        if max_word == answer[b] then
          correct = correct + 1
        end
        num_examples = num_examples + 1

        predictions[b] = max_word
      end
    end

    if saved_model_file then
      local out_dump_file = string.format('%s.%s.%03d.dump', saved_model_file, dump_name, i)
      local out_file = hdf5.open(out_dump_file, 'w')
      local inp = in_words:long()
      local out = outputs:double()
      out_file:write('inputs', inp)
      out_file:write('outputs', out)
      out_file:write('predictions', predictions)
      out_file:write('answers', answer)
      out_file:close()
    end

    if opt.progress then
      xlua.progress(i, ntestbatches)
    end

    if i % opt.gcepoch == 0 then
      collect_track_garbage()
    end
  end

  local accuracy = correct / num_examples
  print('Test Accuracy = '..accuracy..' ('..correct..' out of '..num_examples..')')

  collect_track_garbage()
end

function build_input_embeddings(in_vocab_size, in_post_vocab_size, in_size, in_post_size, in_extr_size, maxseqlen, dropout)
  local emb = nn.Sequential()

  -- input layer (i.e. word embedding space)
  -- input is batchsize x seqlen, output is batchsize x seqlen x insize
  lookup_text = nn.LookupTableMaskZero(in_vocab_size, in_size)
  lookup_post = nn.LookupTableMaskZero(in_post_vocab_size, in_post_size)

  lookup_text.maxnormout = -1 -- prevent weird maxnormout behaviour
  lookup_post.maxnormout = -1

  lookup = nn.Sequential()
    :add(nn.ParallelTable():add(lookup_text):add(lookup_post))
    :add(nn.JoinTable(3)) -- batchsize x seqlen x (insize + in_post_size)

  featurizer = nn.Sequential()
    :add(nn.ParallelTable():add(lookup):add(nn.Mul()))
    :add(nn.JoinTable(3)) -- batchsize x seqlen x (insize + in_post_size + in_extr_size)
    :add(nn.SinusoidPositionEncoding(maxseqlen, in_size + in_post_size + in_extr_size))

  emb:add(featurizer)
  if dropout > 0 then
      emb:add(nn.Dropout(dropout))
  end

  return emb
end

function build_doc_encoder(in_size, batchsize, attstack, atthead, dropout, dff)
  local selfattention = nn.Sequential()

  for i = 1, attstack do
    local encoder = nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.MultiHeadAttention(atthead, in_size, dropout, mask_subsequent = false, single_input = true))
        :add(nn.Identity())) -- batchsize x seqlen x hidsize
      :add(nn.CAddTable())
      :add(nn.LayerNorm(batchsize, in_size))
      :add(nn.ConcatTable()
        :add(nn.PositionWiseFFNN(in_size, dff, dropout))
        :add(nn.Identity())) -- batchsize x seqlen x hidsize
      :add(nn.CAddTable())
      :add(nn.LayerNorm(batchsize, in_size))

    local masked_decoder = nn.Sequential()
      :add(nn.ConcatTable()
        :add(nn.MultiHeadAttention(atthead, in_size, dropout, mask_subsequent = true, single_input = true))
        :add(nn.Identity())) -- batchsize x seqlen x hidsize
      :add(nn.CAddTable())
      :add(nn.LayerNorm(batchsize, in_size))

    selfattention -- input is {QK , V} 
      :add(nn.ConcatTable()
        :add(nn.Sequential() -- encoder-decoder attention
          :add(nn.ParallelTable()
            :add(encoder) -- input embeddings
            :add(masked_decoder)) -- output embeddings
          :add(nn.MultiHeadAttention(atthead, in_size, dropout, mask_subsequent = false, single_input = false)))
        :add(nn.SelectTable(2))) -- V from previous decoder
      :add(nn.CAddTable())
      :add(nn.LayerNorm(batchsize, in_size))
      :add(nn.ConcatTable()
        :add(nn.PositionWiseFFNN(in_size, dff, dropout))
        :add(nn.Identity())) -- batchsize x seqlen x hidsize
      :add(nn.CAddTable())
      :add(nn.LayerNorm(batchsize, in_size))
  end

  selfattention:add(nn.Bottle(nn.Linear(in_size, 1))):add(nn.Squeeze())
  return selfattention
end

function build_model()

  if not lm then
    x_inp = nn.Identity()():annotate({name = 'x', description = 'memories'})

    Embed = build_input_embeddings(vocab_size, post_vocab_size, opt.inputsize, opt.postsize, extr_size, opt.maxseqlen, opt.dropout)
    nng_inp = Embed(x_inp):annotate({name = 'nng_inp', description = 'input embeddings'})

    Shift = nn.Sequential()
      :add(nn.ParallelTable()
        :add(nn.ParallelTable()
            :add(nn.ShiftRight()) -- input text
            :add(nn.ShiftRight())) -- pos tags
        :add(nn.ShiftRight())) -- extra features
      :add(Embed:clone('weight', 'gradWeight', 'bias', 'gradBias'))
    nng_out = Shift(x_inp):annotate({name = 'nng_out', description = 'output embeddings'})

    Yd = build_doc_encoder(opt.inputsize + opt.postsize + extr_size, opt.batchsize, opt.attstack, opt.atthead, opt.dropout, opt.dff)
    nng_Yd = Yd({nng_inp, nng_out}):annotate({name = 'Yd', description = 'memory embeddings'})

    lm = nn.gModule({x_inp}, {nng_Yd})

    -- don't remember previous state between batches since
    -- every example is entirely contained in a batch
    lm:remember('neither')

    if opt.uniform > 0 then
      for k,param in ipairs(lm:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
      end
    end

    -- load pretrained embeddings
    -- IMPORTANT: must do this after random param initialization
    if lookup_text and data.word_embeddings then
      local pretrained_vocab_size = data.word_embeddings:size(1)
      print('Using pre-trained Glove word embeddings')
      lookup_text.weight[{{1, pretrained_vocab_size}}] = data.word_embeddings
    end

    attention_layer = nn.SoftMax() -- to be applied with some mask
  end

  if opt.profile then
    lm:profile()
  end

  if not opt.silent then
    print"Model:"
    print(lm)
  end

  --[[ CUDA ]]--

  if opt.cuda then
    lm:cuda()
    attention_layer:cuda()

    lm = nn.DataParallelTable(2):add(lm, torch.range(1, cutorch.getDeviceCount()):totable()):cuda()
  end

end

function mask_attention(input_context, output_pre_attention)
  -- attention while masking out stopwords, punctuations
  for i = 1, input_context:size(1) do -- batchsize
    for j = 1, input_context:size(2) do -- seqlen
      if input_context[i][j] == 0 or puncs[input_context[i][j]] ~= nil or stopwords[input_context[i][j]] ~= nil then
        output_pre_attention[i][j] = -math.huge
      end
    end
  end
  return attention_layer:forward(output_pre_attention)
end

function mask_attention_gradients(input_context, output_grad)
  -- input_context is batchsize x seqlen
  -- output_grad is batchsize x seqlen
  for i = 1, input_context:size(1) do
    for j = 1, input_context:size(2) do
      if input_context[i][j] == 0 or puncs[input_context[i][j]] ~= nil or stopwords[input_context[i][j]] ~= nil then
        output_grad[i][j] = 0
      end
    end
  end
end

function train(params, grad_params, epoch)
  local num_examples = data.train_location:size(1)
  local all_batches = torch.range(1, num_examples, opt.batchsize)
  local nbatches = all_batches:size(1)
  local randind = epoch == 1 and torch.range(1,nbatches) or torch.randperm(nbatches)
  local grad_outputs = torch.zeros(opt.batchsize, 100) -- preallocate
  if opt.cuda then
    grad_outputs = grad_outputs:cuda()
  end

  print('Total # of batches: ' .. nbatches)

  lm:training()
  local sumErr = 0

  local all_timer = torch.Timer()
  nbatches = opt.maxbatch == -1 and nbatches or math.min(opt.maxbatch, nbatches)
  -- for ir = 1,1 do
  for ir = 1,nbatches do
    local a = torch.Timer()
    local inputs, answers, answer_inds = loadData(data.train_data, data.train_post, data.train_extr, data.train_location, false, all_batches[randind[ir]])
    if opt.profile then
      print('Load training batch: ' .. a:time().real .. 's')
    end

    local function feval(x)
      if x ~= params then
        params:copy(x)
      end
      grad_params:zero()

      -- forward
      local outputs_pre = lm:forward(inputs)
      local outputs = mask_attention(inputs[1][1], outputs_pre)

      if opt.verbose and ir % 10 == 1 and opt.model == 'crf' then
        -- print('inputs')
        -- print(inputs[{{},2}]:contiguous():view(1,-1))
        -- print('targets')
        -- print(targets[1][{{},2}]:contiguous():view(1,-1))
        -- print(targets[2][{{},2}]:contiguous():view(1,-1))
        
        for inode,node in ipairs(lm.forwardnodes) do
          -- if node.data.annotations.name == 'Yd' then
          --   print(node.data.annotations.name)
          --   print(node.data.module.output[{2}])
          --   print(node.data.annotations.name)
          -- end
          -- if node.data.annotations.name == 'u' then
          --   print(node.data.annotations.name)
          --   print(node.data.module.output:squeeze()[{2}]:view(1,-1))
          --   print(node.data.annotations.name)
          -- end
          -- if node.data.annotations.name == 'Joint' then
          --   print(node.data.annotations.name)
          --   print(node.data.module.output:squeeze()[{2}]:view(1,-1))
          --   print(node.data.annotations.name)
          -- end
          if node.data.annotations.name == 'CRF' then
            local crftheta = node.data.module.output
            print('crftheta')
            print(crftheta)
          end
          if node.data.annotations.name == 'Theta' or node.data.annotations.name == 'CRF' then
            local o = node.data.module.output
            print(node.data.annotations.name..' min = '..o:min()..', max = '..o:max()..', avg = '..o:mean())
            -- print(node.data.annotations.name)
            -- print(node.data.module.output[{{},{},1}])
            -- print(node.data.module.output[{{},{},2}])
            -- print(node.data.annotations.name)
          end
          -- if node.data.annotations.name == 'NOM' then
          --   print(node.data.annotations.name)
          --   print(node.data.module.output:squeeze())
          --   print(node.data.annotations.name)
          -- end
        end
      end

      grad_outputs:resize(opt.batchsize, outputs:size(2)):zero()

      -- compute attention sum, loss & gradients
      local a = torch.Timer()
      local err = 0
      for ib = 1, opt.batchsize do
        if answers[ib] > 0 and puncs[answers[ib]] == nil and stopwords[answers[ib]] == nil then -- skip 0-padded examples & stopword/punctuation answers
          local prob_answer = 0
          for ians = 1, #answer_inds[ib] do
            prob_answer = prob_answer + outputs[ib][answer_inds[ib][ians]]
          end
          if prob_answer == 0 then
            print('ERROR: zero prob assigned to the correct answer at the following indices: ')

            for inode,node in ipairs(lm.forwardnodes) do
              if node.data.annotations.name == 'Yd' then
                print('------------------------------------')
                print(node.data.annotations.name)
                print(node.data.module.output[ib])
              end
            end

            print(answer_inds[ib])
            print('ir = '.. ir .. ', all_batches[randind[ir]] = ' .. all_batches[randind[ir]] .. ', ib = ' .. ib)
            print('inputs')
            print(inputs[1][1][{ib, {}}]:contiguous():view(1,-1))
            print('outputs_pre')
            print(outputs_pre[ib]:view(1,-1))
            print('outputs')
            print(outputs[ib]:view(1,-1))
              
            for inode,node in ipairs(lm.forwardnodes) do
              if node.data.annotations.name == 'u' then
                print('------------------------------------')
                print(node.data.annotations.name)
                print(node.data.module.output[ib]:view(1,-1))
              end
            end

            outputs:foo() -- intentionally break
          end
          for ians = 1, #answer_inds[ib] do
            grad_outputs[ib][answer_inds[ib][ians]] = -1 / (opt.batchsize * prob_answer)
          end
          err = err - torch.log(prob_answer)
        end
      end
      if opt.profile then
        print('Compute attention sum: ' .. a:time().real .. 's')
      end
      sumErr = sumErr + err / opt.batchsize

      if opt.verbose then
        print('grad_outputs: min = ' .. grad_outputs:min() .. 
          ', max = ' .. grad_outputs:max() .. 
          ', mean = ' .. grad_outputs:mean() .. 
          ', std = ' .. grad_outputs:std() .. 
          ', nnz = ' .. grad_outputs[grad_outputs:ne(0)]:size(1) .. ' / ' ..  grad_outputs:numel() .. ' ....................................')

        if grad_outputs:mean() ~= grad_outputs:mean() then -- nan value
          print('ir = '.. ir .. ', all_batches[randind[ir]] = ' .. all_batches[randind[ir]])
          print('inputs')
          print(inputs)
          print('outputs')
          print(outputs)

          for inode,node in ipairs(lm.forwardnodes) do
            if node.data.annotations.name == 'IgnoreZero' then
              print('IgnoreZero')
              print(node.data.module.output)
            end
            if node.data.annotations.name == 'Attention' then
              print('Attention')
              print(node.data.module.output)
            end
          end

          outputs:foo() -- break on purpose
        end
      end

      -- backward 
      grad_outputs = attention_layer:backward(outputs_pre, grad_outputs)
      mask_attention_gradients(inputs[1][1], grad_outputs)

      lm:zeroGradParameters()
      lm:backward(inputs, grad_outputs)
      
      -- update
      if opt.cutoff > 0 then
        local norm = lm:gradParamClip(opt.cutoff) -- affects gradParams
        opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end

       return err, grad_params
    end

    local _, loss = optim.adam(feval, params, opt.adamconfig)

    if opt.progress then
      xlua.progress(ir, nbatches)
    end

    if ir % opt.gcepoch == 0 then
      collect_track_garbage()
    end
  end
  
  collect_track_garbage()

  if not opt.silent then
    if opt.meanNorm then
      print("mean gradParam norm", opt.meanNorm)
    end
  end

  if cutorch then cutorch.synchronize() end
  local secs_per_epoch = all_timer:time().real
  local speed = nbatches*opt.batchsize/secs_per_epoch
  print(string.format("Time per epoch: %f s, Speed : %f words/second; %f ms/word", secs_per_epoch, speed, 1000/speed))

  local loss = sumErr/nbatches
  print("Training error : "..loss)

  xplog.trainloss[epoch] = loss
end

function validate(ntrial, epoch)
  local num_examples = data.valid_location:size(1)
  local all_batches = torch.range(1, num_examples, opt.batchsize)
  local nvalbatches = all_batches:size(1) - 1 -- ignore the last batch which contains zero-padded data

  lm:evaluate()
  local sumErr = 0

  -- for i = 1, 1 do
  for i = 1, nvalbatches do
    local valid_con, valid_ans, valid_ans_ind = loadData(data.valid_data, data.valid_post, data.valid_extr, data.valid_location, false, all_batches[i])

    local inputs = valid_con
    local answer_inds = valid_ans_ind
    local outputs = mask_attention(inputs[1][1], lm:forward(inputs))
    local err = 0
    for ib = 1, opt.batchsize do
      if valid_ans[ib] > 0 and puncs[valid_ans[ib]] == nil and stopwords[valid_ans[ib]] == nil then -- skip 0-padded examples & stopword/punctuation answers
        local prob_answer = 0
        for ians = 1, #answer_inds[ib] do
          prob_answer = prob_answer + outputs[ib][answer_inds[ib][ians]]
        end
        err = err - torch.log(prob_answer)
      end
    end
    sumErr = sumErr + err

    if opt.progress then
      xlua.progress(i, nvalbatches)
    end

    if i % opt.gcepoch == 0 then
      collect_track_garbage()
    end
  end

  collect_track_garbage()

  local validloss = sumErr/nvalbatches
  print("Validation error : "..validloss)

  xplog.valloss[epoch] = validloss
  ntrial = ntrial + 1

  -- early-stopping
  if validloss < xplog.minvalloss then
    -- save best version of model
    xplog.minvalloss = validloss
    xplog.epoch = epoch 
    local filename = paths.concat(opt.savepath, opt.id..'.t7')
    
    test_model()

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

  collect_track_garbage()
end

-- Driver code
data = hdf5.open(opt.datafile, 'r'):all()
puncs = tds.hash()
for i = 1, data.punctuations:size(1) do
  puncs[data.punctuations[i]] = 1
end

stopwords = tds.hash()
for i = 1, data.stopwords:size(1) do
  stopwords[data.stopwords[i]] = 1
end

vocab_size = data.vocab_size[1]
post_vocab_size = data.post_vocab_size[1]
extr_size = data.train_extr:size(2)

if #opt.testmodel > 0 then
  print("Processing test set")
  test_model(opt.testmodel)
  print("Processing analysis set")
  test_model(opt.testmodel, 'analysis', data.analysis_data, data.analysis_post, data.analysis_extr, data.analysis_location)
  os.exit()
end

if opt.unittest then
  test_answer_in_context(data.train_data, data.train_location, -1)
  test_answer_in_context(data.valid_data, data.valid_location, -1)
end

if data.word_embeddings then
  opt.inputsize = data.word_embeddings:size(2)
end

if not opt.silent then 
  print("Vocabulary size : "..vocab_size) 
  print("Using batch size of "..opt.batchsize)
end

build_model()

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
if not xplog then
  xplog = {}
  xplog.opt = opt -- save all hyper-parameters and such
  xplog.puncs = puncs
  xplog.dataset = 'Lambada'
  -- will only serialize params
  xplog.model = nn.Serial(lm)
  xplog.model:mediumSerial()
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
local params, grad_params = lm:getParameters()
local adamconfig = {
   beta1 = opt.adamconfig[1],
   beta2 = opt.adamconfig[2],
   learningRate = opt.lr
}

while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
  print("")
  print("Epoch #"..epoch.." :")

  train(params, grad_params, epoch)
  validate(ntrial, epoch)

  epoch = epoch + 1
end

test_model()
