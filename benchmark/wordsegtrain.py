
# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

import re
import os
import pickle
import sys
import nltk
from collections import Counter

f = sys.argv[1]

def tokenize(text):
    return nltk.word_tokenize(text)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
text = open(f).read()

def pairs(iterable):
    iterator = iter(iterable)
    values = [next(iterator)]
    for value in iterator:
        values.append(value)
        yield ' '.join(values)
        del values[0]

save_obj(Counter(tokenize(text)), os.path.basename(f) + "_unigrams.bin")
save_obj(Counter(pairs(tokenize(text))), os.path.basename(f) + "_bigrams.bin")

