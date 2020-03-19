
# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

import codecs
import sys
import kenlm

def spcing(s):
    ss = s.replace(" ", "_")
    return reduce(lambda acc, x: acc + x + "\x20", ss, "").strip() 

def unspcing(s):
    return s.replace(" ", "").replace("_", " ")

def seg(txt, part, n, t, w, p):
    if len(txt) == 0:
        return part
    if len(part) == 0:
        return seg(txt[1:], [txt[0]], n, t, w, p)
    return seg(txt[1:], beam(txt[0], part, n, t, w, p), n, t, w, p)

def beam(c, part, n, t, w, p):
    return topn(xpd(c, part, t, w, p), n, p)

def xpd(c, part, t, w, p):
    if len(part) == 0:
        return []
    return xpd(c, part[1:len(part)], t, w, p) + [part[0] + c] + [bnd(c, part[0], t, w, p)]

def bnd(c, x, t, w, p):
    nxt = x[len(x) - w:] + "\x20" + c
    if p(nxt) > -t:
        return nxt
    return ""

def topn(part, n, p):
    spart = list(part)
    spart.sort(key = lambda x: p(x), reverse = True)
    return spart[0:n]

def r(stng):
    return filter(lambda x: x != "\x20", stng)

def segment(txt, t, n, w, p, m):
    return seg(r(txt), [], t, n, w, p)[0:m]


if __name__ == "__main__":
    reload(sys)  # Reload does the trick! (hack!)
    sys.setdefaultencoding('UTF8')
    sys.setrecursionlimit(3000)
    if len(sys.argv) < 5:
        print("Uso: ")
        sys.exit()
    txt = sys.argv[6].decode("utf-8")
    model = kenlm.Model(sys.argv[1])
    p = lambda x: model.score(spcing(x), bos = True, eos = False)
    t = int(sys.argv[2])
    n = int(sys.argv[3])
    w = int(sys.argv[4])
    m = int(sys.argv[5])
    result = segment(txt, t, n, w, p, m)
    for r in result:
        print(txt + "\t" + r + "\t" + str(p(r)))
    
