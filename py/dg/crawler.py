import collections
import urllib2
import shlex
import tokenscaner
import re

class Crawler:
    def __init__(self):
        pass

    def grabPage(self, url):
        html = urllib2.urlopen(url).read()
        try:
            j = html.find('charset=') + len('charset=')
            i = j
            while html[i] not in [' ', ',', '"', "'", ')', '(', '>', '<']:
                i += 1
            charset = html[j:i]
            print 'charset', charset
            html = unicode(html.decode(charset))
            html = html.encode('utf-8')
        except Exception as e:
            print 'cannot parse charset', url, e
        html = html.lower()
        #print html
        urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', html)
        #urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', html)
        #urls2 = filter(
        #        lambda x: True,#x[0] == '.',
        #        re.findall(r'\./[^\s<>"]+|www\.[^\s<>"]+', unicode(html))
        #        #re.findall(r'\./[^\s<>"]+|www\.[^\s<>"]+', html)
        #    )
        #domain = url[:url.find('/', 3)]
        #urls2 = map(lambda x: domain + x[1:], urls2)
        #re.findall('./(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', html)
        #print 'urls2', urls2
        freq = collections.defaultdict(lambda :0)
        #shl = shlex.shlex(html)
        ts = tokenscaner.TokenScaner(html)
        while not ts.empty():
            freq[ts.getToken()] += 1
        return freq, urls

    def grabFromPage(self, url, depth, walked=None, only_sub_domens=False):
        if depth < 1: return {}
        if walked == None:
            walked = {}
        stack = [url]
        main_url = url
        walked['walked_' + url] = ['0']
        res = collections.defaultdict(lambda : [])
        for i in xrange(depth):
            new_stack = []
            total = 0
            for url in stack:
                total += 1
                if main_url not in url: continue
                nurl = 'walked_' + url
                print nurl
                if nurl in walked and walked[nurl][0] == '1': continue
                print url, total, '/', len(stack), '/', i, '/', depth
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
        return res, stack
                
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
