import json
import udp_common
import threading
import socket
import time

class DHT:
    def __init__(self, boot_addr, boot_port, r_addr, r_port):
        self.boot_addr = boot_addr
        self.boot_port = boot_port
        self.r_addr = r_addr
        self.r_port = r_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        self.sock.bind((self.r_addr, self.r_port))
        self.op_count = 0

    def delay(self):
        time.sleep(0.01)
        #if self.op_count % 5 == 0:
        #    time.sleep(0.005)
        #if self.op_count % 10 == 0:
        #    time.sleep(0.1)

    def __getitem__(self, key):
        self.op_count += 1
        msg = {'method' : 'get', 'key' : key, 'addr' : [self.r_addr, self.r_port]}
        try:
            udp_common.send(self.boot_addr, self.boot_port, json.dumps(msg))
            data, addr = self.sock.recvfrom(65535) # buffer size is 1024 bytes
            ans = json.loads(data)
            value = ''
            while 'next' in ans:
                v += ans['result']
                data, addr = self.sock.recvfrom(65535) # buffer size is 1024 bytes
                ans = json.loads(data)
            value += ans['result']
            self.delay()
            return json.loads(value)
        except Exception as e:
            print 'get item', e
            return None

    def __setitem__(self, key, value):
        self.op_count += 1
        try:
            value = json.dumps(value)
            while len(value) > 40000:
                value_to_send = value[:40000]
                value = value[40000:]
                msg = {'method' : 'set', 'key' : key, 'value' : value_to_send, 'addr' : [self.r_addr, self.r_port], 'next' : 1}
                udp_common.send(self.boot_addr, self.boot_port, json.dumps(msg))
            msg = {'method' : 'set', 'key' : key, 'value' : value, 'addr' : [self.r_addr, self.r_port]}
            udp_common.send(self.boot_addr, self.boot_port, json.dumps(msg))
            data, addr = self.sock.recvfrom(65535) # buffer size is 1024 bytes
            self.delay()
        except Exception as e:
            print 'set item', e
            return None

    def __contains__(self, key):
        return self.__getitem__(key) != None



if __name__ == '__main__':
    print 'Start'
    dht = DHT("127.0.0.1", 6000, "127.0.0.1", 6101)
    print dht['123']
    dht['123'] = 10#json.dumps([10])
    print dht['123']
    print dht['321']
    for i in xrange(1000):
        dht['key' + str(i)] = str(i) * 10000
    for i in xrange(1000):
        print dht['key' + str(i)] == str(i) * 10000
    print '222' in dht

    #t = threading.Thread(target=udp_common.listen,  args=(handler, "127.0.0.1", 6101, lambda : False))
    #def sender():
    #    print 'Start'
    #t.run()
    #t2 = threading.Thread(target=sender, args=())
    #t2.run()
    #print 'Start'
    #print msg
