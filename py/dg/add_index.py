import dg, sys
import dht as pydht
import crawler

if __name__ == '__main__':
    port, url, depth = sys.argv[1:4]
    cr = crawler.Crawler()
    print 'creating dht'
    #dht = pydht.DHT("127.0.0.1",6111, boot_host="127.0.0.1", boot_port=5001)
    print 'grabbing...'
    #freq = cr.grabFromPage('http://habrahabr.ru',2, dht)
    dht = pydht.DHT("127.0.0.1", int(port), "127.0.0.1", 6101)
    print dht['java']
    exit()
    freq, urls = cr.grabFromPage(url, int(depth), {})
    #dht = pydht.DHT("192.168.10.128", int(port), "192.168.10.1", 6101)
    #freq, urls = cr.grabFromPage('http://www.cs.vsu.ru/~svv/lectures.html',3, dht, True)
    print 'adding index'
    dg.shareIndex(dht, freq)
    print urls

