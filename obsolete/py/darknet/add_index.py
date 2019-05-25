import sys
import dht as pydht

if __name__ == '__main__':
    port, url, file = sys.argv[1:4]
    force = len(sys.argv) > 4
    print 'creating dht'
    dht = pydht.DHT("127.0.0.1", int(port), "127.0.0.1", 6100)
    #freq, urls = cr.grabFromPage('http://www.cs.vsu.ru/~svv/lectures.html',3, dht, True)
    if url not in dht or force:
        print str(open(file).read())
        dht[url] = str(open(file).read())
    else:
        print 'url in base'

