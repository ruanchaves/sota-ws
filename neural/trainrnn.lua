
-- Copyright (C) 2017 Yerai Doval 
-- You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

require "torch"
require "rnnlm"
require "lfs"
require "../utyls/utyls.lua"

torch.setdefaulttensortype('torch.CudaTensor')

if #arg < 1 then
	print("Usage: " .. arg[0] .. " <outputModelFile> <outputSize> <trainFile (train.txt)> <evalFile(eval.txt)> <batchSize (120)> <maxIter (100)> <hiddenSizes ({1000,1000})>")
	os.exit()
end

local outputModelFile = arg[1] or "model-out"
local outputSize = tonumber(arg[2])
local trainFile = arg[3] or "train.txt"
local evalFile = arg[4] or "eval.txt"
local batchSize = tonumber(arg[5]) or 120
local maxIter = tonumber(arg[6]) or 100
local stmodel = arg[7] or "0"
local hiddenSizes = {}
for i=8, #arg do
	table.insert(hiddenSizes, tonumber(arg[i]))
end
if #hiddenSizes == 0 then hiddenSizes = {1000, 1000} end

torch.manualSeed(12357)

local net, crit 

if stmodel == "0" then
	net, crit = rnnlm(hiddenSizes, outputSize)
else
	local prevm = torch.load(stmodel)
	net = prevm.net 
	_, crit = rnnlm(hiddenSizes, outputSize)
end

net:cuda()
crit:cuda()

print("#" .. table.tostring(arg):gsub("\n", "\n# "))
print("\n#\n#***\n#\n")
print("#" .. tostring(net):gsub("\n", "\n# "))
print("\n#\n#***\n#\n")

local iter = train(net, trainFile, batchSize, crit)
local j = 0
if trainFile:match(".bin") ~= nil then iter = bintrain(net, trainFile, crit) end
local epochTimer = torch.Timer()
local instTimer = torch.Timer()
for i=1, maxIter do
	epochTimer:reset()
	print("# begin epoch " .. i)
	instTimer:reset()
	for i, trainErr, state, params, gradParams in iter do
		local getInstTimer = instTimer:time().real 
		local evalErr = eval(net, evalFile, batchSize, crit)
		j = j + 1
		local stateFields = "\t"
		for k, v in pairs(state) do
			local sv
			if type(v) == "userdata" then
				sv = v:mean()
			else
				sv = v
			end
			stateFields = stateFields .. sv .. "\t"
		end
		print(j .. "\t" .. trainErr .. "\t" .. evalErr .. stateFields .. params:mean() .. "\t" .. gradParams:mean() .. "\t" .. getInstTimer)
		if j % 10 == 0 then 
			torch.save(outputModelFile .. "-continuation.t7", {net = nn.Serial(net, 'torch.FloatTensor'):mediumSerial(), crit = crit})
			torch.save(outputModelFile .. "-" .. j .. ".t7", {net = nn.Serial(net, 'torch.FloatTensor'):lightSerial(), crit = crit})
		end
		instTimer:reset()
	end
	print("# t = " .. epochTimer:time().real .. " s")
end
