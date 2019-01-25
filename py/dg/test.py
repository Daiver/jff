import threading
import pydht
import db

if __name__ == '__main__':
    main_dht = pydht.DHT("127.0.0.1", 5000)
    sub_dht = pydht.DHT("127.0.0.1", 6000, boot_host="127.0.0.1", boot_port=5000)
    port_pool = map(lambda x: 5000 +x, xrange(1, 10))
    print port_pool
    sub_dht['key'] = ['some_value']
    dht_pool = map(lambda x: pydht.DHT("127.0.0.1", x, boot_host="127.0.0.1", boot_port = 5000, dbw=db.DumpWorker(x)), port_pool)
    #print dht_pool[8]['key']
    for i, d in enumerate(dht_pool):
        data = d.dbw.select()
        print i
        for k, v in data:
            d[k] = v
    print 'ready'
    try:
        raw_input()
    except KeyboardInterrupt as e:
        print 'dumping'
        for i, d in enumerate(dht_pool):
            print i
            d.dump()
    print 'end'
