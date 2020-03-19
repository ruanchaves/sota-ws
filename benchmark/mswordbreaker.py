
# Copyright (C) 2017 Yerai Doval 
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/gpl.txt>

import httplib, urllib, base64, re, string, sys, json
from unidecode import unidecode

rexp = re.compile('[\W_]+', re.UNICODE)
for line in sys.stdin:
    x, _ = line.split("\t")
    x = unidecode(unicode(x, "utf-8"))
    headers = {
        # Request headers
        'Ocp-Apim-Subscription-Key': None, # <-- Insert your API key here
        }
    dx = rexp.sub('', x.lower())
    params = urllib.urlencode({
        # Request parameters
        'model': 'body',
        'text': dx,
        'order': 3,
        'maxNumOfCandidatesReturned': 5,
        })

    try:
        conn = httplib.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request("POST", "/text/weblm/v1.0/breakIntoWords?%s" % params, "{body}", headers)
        response = conn.getresponse()
        data = response.read().decode()
        parsedData = json.loads(data)
        candidatesStr = x
        for cand in parsedData["candidates"]:
            candidatesStr = candidatesStr + "\t" + cand["words"]
        print(candidatesStr)
        conn.close()
    except Exception as e:
        print("[{0}] {1}".format(e.errno, e.strerror))
