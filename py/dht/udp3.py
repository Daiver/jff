import random
import threading
import json

from udp_common import send, listen
import simple_node
import commandh

def build_command_handler(event, pool):
    def command_handler(data, addr):
        if  data == 'reload_source':
            print 'reloading'
            reload(udp3)
            print 'sending'
            request = {
                        'method' : "reload_source",
                        'addr' : "127.0.0.1",
                        'port' : 5000
                    }
            for addr, port in pool:
                send(addr, port, json.dumps(request))
            print 'finish'

        if  data == 'exit':
            print 'closing'
            event.set()
            for addr, port in pool:
                send(addr, port, "exit")
            print 'finish'
    return command_handler

if __name__ == '__main__':

    addr_pool = [
            ("127.0.0.1", 5001),
            ("127.0.0.1", 5002),
            ("127.0.0.1", 5003),
            ("127.0.0.1", 5004),
            ("127.0.0.1", 5005),
        ]

    node_pool = map(lambda x: simple_node.SimpleNode(x[0], x[1], simple_node), addr_pool)
    e1 = threading.Event()
    thread_pool = map(lambda x: threading.Thread(target=listen, args=(x.handle, x.addr, x.port, e1.isSet)), node_pool)

    t1 = threading.Thread(target=listen, args=(commandh.build_command_handler(e1, addr_pool, commandh), "127.0.0.1", 5000, e1.isSet))
    t1.start()
    for t in thread_pool: t.start()
    for a1, p1 in addr_pool:
        for a2, p2 in addr_pool:
            send(a1, p1, '{"method":"connect", "target_addr":"%s", "target_port" : %s}' % (a2, p2))
    import sha
    #node_pool[0].content_table[sha.new('123').hexdigits()] = 'YEAH'
    for t in thread_pool: t.join()
