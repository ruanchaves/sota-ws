
-- Copyright (C) 2017 Yerai Doval 
-- You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

require 'torch'
require 'lfs'
local fun = require 'fun' 
require 'rnnlm' 

function spairs(t, order)
	local keys = {}
	for k in pairs(t) do keys[#keys+1] = k end
	if order then
		table.sort(keys, function(a,b) return order(t, a, b) end)
	else
		table.sort(keys)
	end
	local i = 0
	return function()
		i = i + 1
		if keys[i] then
			return keys[i], t[keys[i]]
		end
	end
end

function compare(t, a, b)
	return t[b][1] < t[a][1]
end

function bestcand(cands, n)
	local result = {}
	local c = 1
	for k, v in spairs(cands, compare) do
		if c > n then
			break
		end
		result[k] = v
		c = c + 1
	end

	return result
end

function sort_order(a, b)
	return a[2] > b[2]
end

function getSpcProbs(logProbs, toScore)
	local spcProbs = {}
	local spcChar = " "
	local spcIdx = spcChar:byte() - 31
	local lastLogProbs = nil
	local batchSize = nil
	if type(logProbs) == "table" then
		lastLogProbs = logProbs[#logProbs]
		batchSize = logProbs[1]:size(1)
	else
		lastLogProbs = logProbs[-1]
		batchSize = logProbs:size(2)
	end
	for i=1, batchSize do
		spcProbs[toScore[i]] = lastLogProbs[i][spcIdx]
	end
	return spcProbs
end

function segment(text, nn, threshold, win, n)
	text = text:gsub(" ", "")
	local output_list = {{"", -math.huge}}
	local spc = " "
	local spc_b = spc:byte() - 31
	local prev_iterator = text:gmatch(".")
	local position = 1
	local spc_probs = {}
	local threshold = -threshold

	for prev_c in prev_iterator do
		local partial = {}
		local toScore = {}
		position = position + #prev_c
		for i=1, #output_list do
			local output = output_list[i][1]
			local spcposition = position + #output:gsub("[^%s]+", "")
			output = output .. prev_c
			local spc_prob = spc_probs[" " .. output .. " "]
			local lim_sup = position + 1
			local lim_inf = position - win
			if lim_inf < 1 then
				lim_inf = 1
			end
			while output:sub(lim_inf,lim_inf) ~= " " and lim_inf > 1 do
				lim_inf = lim_inf - 1
			end
			if spc_prob and spc_prob > threshold or i == 1 then
				local w_spc1 = output:sub(lim_inf, spcposition - 1) 
				.. " " .. text:sub(position, position)
				local ts1 = ("\n" .. w_spc1 .. "\n"):gsub("%s+", " ")
				table.insert(toScore, ts1)
				table.insert(partial, {output .. " "})
			end
			local wo_spc1 = output:sub(lim_inf, spcposition - 1) .. text:sub(position, position)
			local ts2 = ("\n" .. wo_spc1 .. "\n"):gsub("%s+", " ")
			table.insert(toScore, ts2)
			table.insert(partial, {output})
		end
		local scores, logProbs = scoreStrings(toScore, nn)
		spc_probs = getSpcProbs(logProbs, toScore)
		local lscores = 0
		if type(scores) == "table" then
			lscores = #scores
		else
			lscores = scores:size(1)
		end
		for i=1,lscores do
			table.insert(partial[i], scores[i])
		end
		output_list = fun.totable(fun.chain(output_list, partial))
		local tmp = {}
		for i=1,#output_list do
			local no_spc = output_list[i][1]:gsub("%s+", "")
			if #no_spc >= position - 1 then
				table.insert(tmp, output_list[i])
			end
		end
		output_list = tmp
		table.sort(output_list, sort_order)
		if #output_list > n then
			local n_best = {}
			for i=1,n do
				table.insert(n_best, output_list[i])
			end
			output_list = n_best
		end
	end
	return output_list
end

cmd = torch.CmdLine()
cmd:argument('-text','text to tokenize.')
cmd:option('-model', 'model.t7','model checkpoint to use for sampling')
cmd:option('-threshold',8,'')
cmd:option('-prun',5,'')
cmd:option('-win',64,'')
cmd:option('-nresults',1,'')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:text()

opt = cmd:parse(arg)

function gprint(str)
	if opt.verbose == 1 then print(str) end
end

torch.manualSeed(opt.seed)

if not lfs.attributes(opt.model, 'mode') then
	gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
local checkpoint = torch.load(opt.model)
local nn = checkpoint.net:cuda()

local text = opt.text
local t2 = text:gsub("^%s*(.-)%s*$", "%1")
local result = segment(t2, nn, opt.threshold, opt.win, opt.prun)
local candSet = {}

local c = 1
for _,v in spairs(result, compare) do
	if c > opt.nresults then
		break
	end
	local cand = v[1]:gsub("^%s*(.-)%s*$", "%1")
	if not candSet[cand] then
		print(text .. "\t" .. cand .. "\t" .. v[2])
		candSet[cand] = true
		c = c + 1
	end
end
