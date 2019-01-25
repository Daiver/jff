import collections
import urllib2
import shlex
import tokenscaner
import re

class Crawler:
    def __init__(self):
        pass

    def grabPage(self, url):
        html = urllib2.urlopen(url).read().lower()
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', html)
        freq = collections.defaultdict(lambda :0)
        #shl = shlex.shlex(html)
        ts = tokenscaner.TokenScaner(html)
        while not ts.empty():
            freq[ts.getToken()] += 1
        return freq, urls

    def grabFromPage(self, url, depth, walked=None):
        if depth < 1: return {}
        if walked == None:
            walked = {}
        stack = [url]
        res = collections.defaultdict(lambda : [])
        for i in xrange(depth):
            new_stack = []
            for url in stack:
                nurl = 'walked_ ' + url
                if nurl in walked and walked[nurl][0] == '1': continue
                print url
                try:
                    freq, urls = self.grabPage(url)
                    walked[nurl] = ['1']
                    urls = filter(lambda x:x[-4:] not in ['.jpg', '.png'], urls)
                    new_stack += urls
                    for k, v in freq.iteritems():
                        res[k] += [(url, v)]
                except Exception as e:
                    print 'EXCP', url, '\n', e
            stack = new_stack
        return res
                
        #for x in shl:
        #    freq[x.encode('UTF8')] += 1
        #for x in freq.keys():
        #    print (x)
        #print urls

if __name__ == '__main__':
    crawler = Crawler()
    #crawler.grabPage('http://habrahabr.ru')
    res = crawler.grabFromPage('http://habrahabr.ru', 2)
    for k, v in res.iteritems():
        if len(v) > 1:
            print k, v
