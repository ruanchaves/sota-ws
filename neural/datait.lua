
-- Copyright (C) 2017 Yerai Doval 
-- You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

local fun = require "fun"
require 'cutorch'

function adapt(riter)
	return function()
		local v = nil
		pcall(function() v = riter() end)
		return v, v
	end
end

function wordToIdx(word)
	if word == nil then return nil end
	local idx = {}
	for c in word:gmatch(".") do
		table.insert(idx, math.max(0, c:byte() - 31))
	end
	return idx
end

function getMaxLength(tab)
	return fun.reduce(
	function(max, x)
		if #x > max then return #x else return max end
	end, 0, tab)
end

function padBatch(t, maxLength)
	return fun.totable(fun.map(
	function(x)
		return fun.totable(fun.chain(fun.take(maxLength - #x, fun.zeros()), x)) 
	end, t))
end

function strjoin(delimiter, list)
	local len = #list
	if len == 0 then
		return ""
	end
	local string = list[1]
	for i = 2, len do
		string = string .. delimiter .. list[i]
	end
	return string
end

function nByteChunkIter(fd, chunkId, seqLength, nseq)
	local seqLength = seqLength or 32
	local last = "\n" 
	local i = 1
	fd:seek("set", seqLength*nseq*chunkId) -- avanzamos a la posicion deseada en el texto
	while fd:read(1) ~= "\n" do end -- avanzamos al siguiente comienzo de contexto
	return function()
		if i > nseq then 
			fd:close()
			return nil 
		end 
		i = i + 1
		local text = last .. fd:read(seqLength)
		if #text == 0 then 
			fd:close()
			return nil 
		end		
		local x = text:sub(1, #text - 1)
		local y = text:sub(2, #text) 
		last = text:sub(#text, #text)
		return wordToIdx(x), wordToIdx(y)
	end
end

function getTextSize(byteIter)
	return fun.reduce(function(acc, x) return acc + #x end, 0, byteIter)	
end

function nByteChunkIters(filename, seqLength, batchSize)
	local nseq = math.floor(getTextSize(adapt(io.lines(filename))) / (batchSize * seqLength))
	return fun.totable(fun.map(function(x) return nByteChunkIter(io.open(filename), x, seqLength, nseq) end, fun.range(0, batchSize - 1)))
end

function batchChunkIter(chunkIters)
	return function()
		local xs = {}
		local ys = {}
		for _, chunkIter in pairs(chunkIters) do
			local x, y = chunkIter()
			table.insert(xs, x)
			table.insert(ys, y)
		end
		if #xs == 0 or #ys == 0 then
			return nil, nil 
		end

		local maxLength = getMaxLength(ys)
		xs = padBatch(xs, maxLength)
		ys = padBatch(ys, maxLength)
		return xs, ys
	end
end

function toTensor(iter)
	return fun.map(function(x) return torch.CudaTensor(x) end, iter)
end

function prepareBatch(corpusFile, batchSize)
	local fd = io.open(corpusFile)
	local batches = batchChunkIter(nByteChunkIters(corpusFile, 50, batchSize)) --TODO: seqLength param!!!!
	return function()
		local xs, ys = batches()
		if xs == nil or ys == nil then 
			fd:close()
			return nil, nil
		end
		local zx = fun.map(function(...) return {...} end, fun.zip(unpack(xs)))
		local zy = fun.map(function(...) return {...} end, fun.zip(unpack(ys)))
		return torch.Tensor(fun.totable(zx)), torch.Tensor(fun.totable(zy))
	end
end
