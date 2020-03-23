
-- Copyright (C) 2017 Yerai Doval 
-- You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

require 'torch'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'dpnn'
require 'nngraph'
require 'datait'
local fun = require 'fun'
-- require 'yo'

function nextByteLogProb(str, rnn)  
	rnn:evaluate()
	rnn:forget()
	local idxs = wordToIdx(str)
	local input = {}
	for _, idx in pairs(idxs) do
		table.insert(input, {idx})
	end
	local logProbs = rnn:forward(torch.CudaTensor(input))
	return logProbs[#idxs]
end

function byteLogProb(b, str, rnn)
	rnn:evaluate()
	local bnum
	if type(b) == number then bnum = b else bnum = b:byte() end
	local logProbs = nextByteLogProb(str, rnn)
	return logProbs[1][bnum]
end

function getBestLogCands(n, str, rnn) 
	rnn:evaluate()
	local logProbs = nextByteLogProb(str, rnn)
	local nxtProbs = logProbs[#logProbs]
	local sorted, oldidxs = nxtProbs:abs():sort() 
	local result = {}
	for i = 1, n do
		table.insert(result, {oldidxs[1][i], sorted[1][i]})
	end
	return result 
end

function getTensorMaxLength(tab)
	return fun.reduce(
	function(max, x)
		if x:size(1) > max then return x:size(1) else return max end
	end, 0, tab)
end

function padBatch(t, maxLength)
	return fun.totable(fun.map(
	function(x)
		return fun.totable(fun.chain(fun.take(maxLength - #x, fun.zeros()), x))
	end, t))
end

function tcopy(obj, seen)
	if type(obj) ~= 'table' then return obj end
	if seen and seen[obj] then return seen[obj] end
	local s = seen or {}
	local res = setmetatable({}, getmetatable(obj))
	s[obj] = res
	for k, v in pairs(obj) do res[tcopy(k, s)] = tcopy(v, s) end
	return res
end

function padTensorBatch(t, maxLength)
	local result = torch.ByteTensor(#t, maxLength):zero()
	for i=1,result:size(1) do
		for j=1,t[i]:size(1) do
			result[i][j] = t[i][j]
		end
	end
	return result
end

function scoreStrings(strs, rnn, prevScores) 
	rnn:evaluate()
	rnn:forget()
	local prevScores = prevScores or fun.totable(fun.take(#strs, fun.zeros()))
	local idxss = fun.totable(fun.map(wordToIdx, strs))
	local maxLength = getMaxLength(idxss)
	local batch = padBatch(idxss, maxLength)
	local x = torch.CudaTensor(batch):t()
	local logProbs = rnn:forward(x)
	local logProbsSize 
	if type(logProbs) == "table" then
		logProbsSize = #logProbs
	else
		logProbsSize = logProbs:size(1)
	end
	local acumProbs = fun.totable(fun.take(#batch, fun.replicate(0)))
	local scores = {}
	local zeros = 0
	for i=1,#batch do
		for j=1, logProbsSize-1 do
			if batch[i][j+1] == 0 then
				zeros = zeros + 1
			else
				acumProbs[i] = acumProbs[i] + logProbs[j][i][batch[i][j+1]]
			end
		end
		scores[i] = acumProbs[i] / (#idxss[i]) + prevScores[i]
	end
	rnn:forget()
	return scores, logProbs
end

function scoreStr(str, rnn)
	rnn:evaluate()
	rnn:forget()
	local idxs = wordToIdx(str)
	local acumProb = 0
	local logProbs = rnn:forward(idxs:view(-1,1))
	local acumProb = 0
	for i=1,#logProbs-1 do
		acumProb = acumProb + logProbs[i][1][idxs[i+1]]
	end
	local score = acumProb / #str

	return score
end

function gradUpdate(rnn, x, y, criterion, params, gradParams, config, state)
	local function feval(p)
		local pred = rnn:forward(x)
		local err = criterion:forward(pred, y)
		local gradCriterion = criterion:backward(pred, y)
		rnn:zeroGradParameters()
		rnn:backward(x, gradCriterion) 
		gradParams:clamp(-5, 5)
		return err, gradParams
	end

	local _, err = optim.adam(feval, params, config, state)
	return err[1]
end

function eval(rnn, evalFile, batchSize, criterion)
	local totalEvalErr = 0
	local i = 0
	rnn:evaluate()
	rnn:forget()
	for x, y in prepareBatch(evalFile, batchSize) do
		i = i + 1
		local pred = rnn:forward(x)
		totalEvalErr = totalEvalErr + criterion:forward(pred, y) / x:size(1)
	end	
	rnn:forget()
	return totalEvalErr / i
end	

function train(rnn, trainFile, batchSize, criterion)
	local i = 0
	local config, state = {}, {} 
	config.learningRate = 0.0001 
	config.learningRateDecay = 1e-8
	local trainIterator = prepareBatch(trainFile, batchSize)
	local flatParams, flatGradParams = rnn:getParameters()
	return function()
		local totalTrainErr = 0
		rnn:training()
		for x, y in trainIterator do
			local err = gradUpdate(rnn, x, y, criterion, flatParams, flatGradParams, config, state)
			i = i + 1
			totalTrainErr = totalTrainErr + err / x:size(1)
			if i % 100 == 0 then
				local trainErr = totalTrainErr / 100 
				collectgarbage()
				return i, trainErr, state, flatParams, flatGradParams
			end
		end
		return nil
	end
end

function rnnlm(hiddenSizes, outputSize)
	local seq = nn.Sequential()
	:add(nn.Sequencer(nn.MaskZero(nn.Sequential()
	:add(nn.BatchNormalization(hiddenSizes[1])), 1)))
	for i=1,#hiddenSizes-1 do
		local seqlstm = nn.SeqLSTM(hiddenSizes[i], hiddenSizes[i+1]) 
		seqlstm.maskzero = true
		seq:add(seqlstm)
		:add(nn.Sequencer(nn.MaskZero(nn.Sequential()
		:add(nn.BatchNormalization(hiddenSizes[i+1])), 1)))
	end

	local lookup = nn.LookupTableMaskZero(224, hiddenSizes[1]) 
	local rnn = nn.Sequential()
	:add(lookup)
	:add(seq)
	:add(nn.Sequencer(nn.MaskZero(nn.Sequential()
	:add(nn.Linear(hiddenSizes[#hiddenSizes], outputSize))
	:add(nn.LogSoftMax()), 1))):remember() 
	local crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1)) 
	return rnn:cuda()
	, crit:cuda()
end
