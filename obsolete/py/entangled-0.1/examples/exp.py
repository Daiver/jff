#! /usr/bin/env python
#
# This library is free software, distributed under the terms of
# the GNU Lesser General Public License Version 3, or any later version.
# See the COPYING file included in this archive
#

import sys
import math

from twisted.protocols import basic
#import entangled.kademlia.protocol
#gtk2reactor.install()
import twisted.internet.reactor
from twisted.internet import stdio
import entangled.dtuple

import entangled.kademlia.contact
import entangled.kademlia.msgtypes

import hashlib

import dg
import crawler

class InputHandler(basic.LineReceiver):
    from os import linesep as delimiter

    def __init__(self, node):
        #super(InputHandler, self).__init__()
        #basic.LineReceiver.__init__(self)
        self.node = node

    def getValue(self, key):
        key = key
        h = hashlib.sha1()
        h.update(key.encode('utf-8'))
        hKey = h.digest()
        def error(failure):
            self.sendLine('ERROR')
        def echo(res):
            if type({}) == type(res):
                self.sendLine('val ' + str(res[hKey]))
            else:
                self.sendLine('no key ' + str(type(res)))

        df = self.node.iterativeFindValue(hKey)
        #df.addCallback(echo)
        df.addErrback(error)
        return df, hKey

    def setValue(self, key, value):
        h = hashlib.sha1()
        h.update(key.encode('utf-8'))
        hKey = h.digest()
        df = self.node.iterativeStore(hKey, value)
        def error(failure):
            self.sendLine('ERROR')
        df.addErrback(error)
        #df.addCallback(self.echo)
        return df

    def massTest(self):
        import time
        st = time.time()
        for i, (x, y) in enumerate(zip(map(lambda x: str(x*10), xrange(10000)) , map(str, xrange(10000)))):
            print x, y
            #if i % 10 == 0:
            #    time.sleep(1)
            self.setValue(x, y)
        print 'ELAPSED'
        print time.time() - st
        time.sleep(2)

    def massCheck(self):
        for x in map(lambda x: str(x*10), xrange(10000)):
            self.getValue(x)

    def connectionMade(self):
        self.transport.write('>>> ')

    def echo(self, res):
        self.sendLine('yep: ' + str(res))
        self.transport.write('>>> ')
        
    def lineReceived(self, line):
        self.sendLine('Echo: ' + line)
        if line == 'test':
            self.massTest()
        if line == 'check':
            self.massCheck()
        if line == 'index':
            cr = crawler.Crawler()
            freq = cr.grabFromPage('http://habrahabr.ru',1)
            dg.shareIndex2(self, freq)
        if 'search' in line:
            q = line[len('search'):]
            dg.activate2(self,q)
        #self.getValue('123')
        #self.getValue('key')
        #self.getValue('123')
        self.transport.write('>>> ')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage:\n%s UDP_PORT  [KNOWN_NODE_IP  KNOWN_NODE_PORT]' % sys.argv[0]
        print 'or:\n%s UDP_PORT  [FILE_WITH_KNOWN_NODES]' % sys.argv[0]
        print '\nIf a file is specified, it should containg one IP address and UDP port\nper line, seperated by a space.'
        sys.exit(1)
    try:
        int(sys.argv[1])
    except ValueError:
        print '\nUDP_PORT must be an integer value.\n'
        print 'Usage:\n%s UDP_PORT  [KNOWN_NODE_IP  KNOWN_NODE_PORT]' % sys.argv[0]
        print 'or:\n%s UDP_PORT  [FILE_WITH_KNOWN_NODES]' % sys.argv[0]
        print '\nIf a file is specified, it should contain one IP address and UDP port\nper line, seperated by a space.'
        sys.exit(1)
    
    if len(sys.argv) == 4:
        knownNodes = [(sys.argv[2], int(sys.argv[3]))]
    elif len(sys.argv) == 3:
        knownNodes = []
        f = open(sys.argv[2], 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            ipAddress, udpPort = line.split()
            knownNodes.append((ipAddress, int(udpPort)))
    else:
        knownNodes = None

    node = entangled.dtuple.DistributedTupleSpacePeer( udpPort=int(sys.argv[1]) )
    
    #window = EntangledViewerWindow(node)
    #window.set_default_size(640, 640)
    #window.set_title('Entangled Viewer - DHT on port %s' % sys.argv[1])
    #window.present()
    
    node.joinNetwork(knownNodes)
    ihandler = InputHandler(node)
    stdio.StandardIO(ihandler)

    import udp_common
    import threading
    import json
    import socket
    def handler(UDP_IP, UDP_PORT):
        print "in listen"
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        sock.bind((UDP_IP, UDP_PORT))
        while True:
            msg, addr = sock.recvfrom(65535)
            def buildGetCallback(hKey, addr):
                def getCallback(ans):
                    if type({}) == type(ans):
                        #if len(ans[hKey]) < 100:
                        #    print ans[hKey]
                        #else :
                        #    print 'Big value'
                        res = ans[hKey]
                        while len(res) > 40000:
                            msg = {'result' : res[:40000], 'next' : 1}
                            res = res[40000:]
                            udp_common.send(addr[0], addr[1], msg)
                            #if len(ans[hKey]) > 40000:
                            #udp_common.send(addr[0], addr[1], str(json.dumps('Too Long')))
                            #else:
                            #    udp_common.send(addr[0], addr[1], str(ans[hKey]))
                        msg = {'result' : res[:40000]}
                        udp_common.send(addr[0], addr[1], json.dumps(msg))
                    else:
                        udp_common.send(addr[0], addr[1], json.dumps({'result' : 'null'}))
                return getCallback

            def buildSetCallback(addr):
                def setCallback(ans):
                    udp_common.send(addr[0], addr[1], str(ans))
                return setCallback


            request = json.loads(msg)
            #print addr, node, request['method'], request['key']
            if request['method'] == 'set':
                #print 'setting....'
                key, value = request['key'], request['value']
                while 'next' in request:
                    msg, addr = sock.recvfrom(65535)
                    request = json.loads(msg)
                    value += request['value']                    

                df = ihandler.setValue(key, value)
                df.addCallback(buildSetCallback(request['addr']))
            if request['method'] == 'get':
                df, hKey = ihandler.getValue(request['key'])
                df.addCallback(buildGetCallback(hKey, request['addr']))

            #print addr, node, request

    t = threading.Thread(target=handler, args=("127.0.0.1", 6000))
    t.run()
    print 'Starting...'

    twisted.internet.reactor.run()
    print 'end'

'''
    def storeValue(self, sender, keyFunc, valueFunc):
        key = 'key'
        h = hashlib.sha1()
        h.update(key)
        hKey = h.digest()
        value = "value"
        df = self.node.iterativeStore(hKey, value)
        df.addCallback(completed)
 
    def deleteValue(self, sender, keyFunc):
        key = keyFunc()
        
        h = hashlib.sha1()
        h.update(key)
        hKey = h.digest()
        self.viewer.msgCounter = 0
        self.viewer.printMsgCount = False
        def completed(result):
            self.viewer.printMsgCount = True
        df = self.node.iterativeDelete(hKey)
        df.addCallback(completed)
 
'''
