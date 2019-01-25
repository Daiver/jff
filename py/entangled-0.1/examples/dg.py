import crawler
import collections
import sys
import time
import json

def activate(dct, query):
    key_words = query.lower().split()
    score = collections.defaultdict(lambda : 0)
    for word in key_words:
        dt = None
        try:
            dt = dct[word]
        except KeyError as e:
            print 'Not found in index:', word
            dt = []

        for u, v, in dt:
            score[u] += v
    #res = map(lambda x:(x[1],x[0]), score.items())
    it = score.items()
    it.sort()
    return it[-10:]

def shareIndex(dht, freq):
    print 'freq len',len(freq)
    def check(k, v):
        try:
            x = dht[k]
            '''if x != v:
                print 'check again'
                print x, v
                check(k, v)'''
        except KeyError as e:
            print 'check again'
            print k, v
            dht[k] = v
            time.sleep(1)
            check(k, v)
    for k, v in freq.iteritems():
        old = None
        '''try:
            old = dht[k]
        except Exception as e:
            print 'ERR1', k, e
            old = []
        v += old'''
        if k:
            #print k
            dht[k] = v
    print 'Checking...'
    for k, v in freq.iteritems():
        if k:
            check(k, v)
            
def shareIndex2(handler, freq):
    for k, v in freq.iteritems():
        handler.setValue(str(k), json.dumps(v))

def activate2(node, query):
    key_words = query.lower().split()
    print key_words
    score = collections.defaultdict(lambda : 0)
    def recur(ans, data):
        hKey, words, score = data
        print ans
        if type(ans) == type({}):
            dt = json.loads(ans[hKey])
            for u, v, in dt:
                score[u] += v
            #res = map(lambda x:(x[1],x[0]), score.items())
            if len(words) > 0:
                df, hKey = node.getValue(words[0])
                df.addCallback(recur, (hKey, words[1:], score))
            else:
                it = score.items()
                it.sort()
                print it[-10:]
                #return it[-10:]
        else:
            print 'BAD WORD'
    df, hKey = node.getValue(key_words[0])
    df.addCallback(recur, (hKey, key_words[1:], score))
    #recur((key_words, score))

if __name__ == '__main__':
    cr = crawler.Crawler()
    print 'creating dht'
    dht = pydht.DHT("127.0.0.1",6111, boot_host="127.0.0.1", boot_port=5001)
    print 'grabbing...'
    freq = cr.grabFromPage('http://habrahabr.ru',3, dht)
    print 'adding index'
    shareIndex(dht, freq)
    while 1:
        s = raw_input('>')
        #print activate(freq, s)
        print activate(dht, s)
