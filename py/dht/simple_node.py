import random
import time
import threading
import json
import sha
import string
from udp_common import send, listen

class SimpleNode:
    def __init__(self, addr, port, module):
        self.port = port
        self.addr = addr
        self.module = module
        self.nodeID = self.getID()
        self.route_table = {}
        self.request_table = {}
        self.content_table = {}
        self.methods = {
                'connect' : self.m_connect,
                'reload_source' : self.m_reload_source,
                'new_route' : self.m_new_route,
                'new_node_connect' : self.m_new_node_connect,
                'print_route_table' : self.m_print_route_table,
                'return' : self.m_return,
                'get' : self.m_get,
                'set' : self.m_set,
            }

    def compareKeys(self, K1, K2):
        res = 0
        for x,y in zip(K1, K2):
            if x == y: res += 1
        return res

    def getID(self):
        N = 40
        return ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(N))

    def sendRequest(self, addr, port, request):
        request.update({'addr' : self.addr, 'port' : self.port, 'nodeID' : self.nodeID})
        send(addr, port, json.dumps(request))

    def m_set(self, request):
        best = max(self.route_table.keys() + [self.nodeID], key=lambda x: self.compareKeys(x, request['request_key']))
        if best == self.nodeID:
            self.content_table[request['request_key']] = request['value']
        else:
            self.sendRequest(self.route_table[best][0], self.route_table[best][1], request)

    def m_get(self, request):
        if request['request_key'] in self.content_table:
            self.sendRequest(request['addr'], request['port'], {
                    'method' : 'return',
                    'value'  : self.content_table[request['request_key']],
                    'request_key' : request['request_key']
                })
            return
        if 'addr' in request:
            self.request_table[request['request_key']] = (request['addr'], request['port'])
        else:
            self.request_table[request['request_key']] = ()
        best = max(self.route_table.keys(), key=lambda x: self.compareKeys(x, request['request_key']))
        if best == self.nodeID:
            if 'addr' in request:
                self.sendRequest(request['addr'], request['port'], {
                        'method' : 'return',
                        'value'  : None,
                        'request_key' : request['request_key']
                    })
            else:
                print self.port, "No value"
        else:
            self.sendRequest(self.route_table[best][0], self.route_table[best][1], request)

    def m_return(self, request):
        if request['request_key'] in self.request_table and len(self.request_table[request['request_key']]):
            a, p = self.request_table[request['request_key']]
            del self.request_table[request['request_key']]
            self.sendRequest(a, p, request)
        else:
            print "ANSEWER", request['value']

    def m_reload_source(self, request):
        import simple_node
        reload(simple_node)
        print self.module
        self.__class__ = simple_node.SimpleNode
        print 'reloading....'

    def m_new_node_connect(self, request):
        self.m_new_route(request)
        routes_to_send = []
        for i in xrange(2):
            routes_to_send.append(random.choice(self.route_table.items()))
        self.sendRequest(request['addr'], request['port'], {
                'method' : 'new_route',
                'additional_routes' : routes_to_send,
            })
        #time.sleep(1)
        #print self.port, self.route_table
        '''routes_to_send = []
        keys = self.route_table.keys()
        if self.nodeID in keys: del keys[self.nodeID]
        if request['nodeID'] in keys: del keys[request['nodeID']]

        max_routes_to_send = 5
        while len(routes_to_send) < 5:
            if len(keys) > 0: break
            k = keys'''

    def m_print_route_table(self, request):
        print self.route_table

    def m_new_route(self, request):
        tmp = request['addr'], request['port']
        id = request['nodeID']
        self.route_table[id] = tmp
        if 'additional_routes' in request:
            for id, tmp in request['additional_routes']:
                self.route_table[id] = tmp


    def m_connect(self, request):
        self.sendRequest(request['target_addr'], request['target_port'], {
                'method' : 'new_node_connect'
            })
        #tmp = request['target_addr'], request['target_port']
        #self.route_table[tmp]

    def handle(self, data, addr):
        #try:
        request = json.loads(data)
        print self.port, ">", request
        self.methods[request['method']](request)
        #except Exception as e:
        #    print "Exception!", data, self.port, "~>", e


