
# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

import wordsegment as ws
import nltk
import pickle
import sys

def identity(s):
    return s

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

if len(sys.argv) == 3: 
    ws.UNIGRAMS = load_obj(sys.argv[1])
    ws.BIGRAMS = load_obj(sys.argv[2])
    ws.TOTAL = float(sum(ws.UNIGRAMS.values()))

ws.clean = identity

for line in sys.stdin:
    line = line.replace("\n", "")
    seg = " ".join(ws.segment(line))
    print(line + "\t" + " ".join(nltk.word_tokenize(seg)).replace("``", "\"").replace("\'\'", "\"") + "\t N/A")
