#import pydht
import dht as pydht
import crawler
import collections
import sys
import time

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
        if dt == None: dt = []
        for u, v, in dt:

            score[u] += v
    it = map(lambda x:(x[1],x[0]), score.items())
    #it = score.items()
    it.sort()
    return it[:-20:-1]

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
        if len(v) == 0: continue
        if k :
            if k in dht:
                print 'updating...'
                old = dht[k]
                #print old
                try:
                    old_dict = {u:c for u, c in old}
                    new_dict = {u:c for u, c in v}
                    old_dict.update(new_dict)
                    v = old_dict.items()
                except Exception as e:
                    print 'bad old', e
                    print old
            #print v
            dht[k] = v
            

        '''try:
            old = dht[k]
        except Exception as e:
            print 'ERR1', k, e
            old = []
        v += old'''
        #if k:
        #print k
        #    dht[k] = v
    print 'Checking...'
    #for k, v in freq.iteritems():
    #    if k:
    #        check(k, v)
            
if __name__ == '__main__':
    cr = crawler.Crawler()
    print 'creating dht'
    #dht = pydht.DHT("127.0.0.1",6111, boot_host="127.0.0.1", boot_port=5001)
    dht = pydht.DHT("127.0.0.1", 6000, "127.0.0.1", 6101)
    print 'grabbing...'
    #freq = cr.grabFromPage('http://habrahabr.ru',2, dht)
    #freq, urls = cr.grabFromPage('http://www.cs.vsu.ru/~svv/',3, {}, True)
    #freq, urls = cr.grabFromPage('http://www.cs.vsu.ru/~svv/lectures.html',3, dht, True)
    print 'adding index'
    #shareIndex(dht, freq)
    #print urls
    while 1:
        s = raw_input('>')
        #print activate(freq, s)
        print activate(dht, s)
