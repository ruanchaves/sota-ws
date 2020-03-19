
# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

from unidecode import unidecode
import sys

if __name__ == "__main__":
    for line in sys.stdin:
        sys.stdout.write(unidecode(unicode(line, "utf-8")))

