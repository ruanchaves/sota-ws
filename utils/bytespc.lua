
-- Copyright (C) 2017 Yerai Doval 
-- You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

for line in io.lines() do
	local s = line:gsub(" ", "_")
	local r = s:gsub("(.)", "%1 ")
	local q = r:gsub(" $", "")
	print(q)
end
