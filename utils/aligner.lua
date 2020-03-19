
-- Copyright (C) 2017 Yerai Doval 
-- You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

for line in io.lines() do
	local pad = ("\0"):rep(arg[1])
	print(pad .. "\n" .. line)
end
