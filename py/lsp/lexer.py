import tokenize
from StringIO import StringIO

def getTokens(text):
    res = []
    tokens = (tokenize.generate_tokens(StringIO(text).readline))
    return list(tokens)
#    for token in tokens:
        #if token[1].find('(') > -1:
            #tmp = token[1].split('(')
            #for x in tmp:
                #res.append((token[0], x, token[2], token[3], token[4])

