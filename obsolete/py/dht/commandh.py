import threading
import json
from udp_common import send, listen

def build_command_handler(event, pool, module):
    def command_handler(data, addr):
        if  data == 'reload_source':
            print 'reloading'
            reload(module)
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

