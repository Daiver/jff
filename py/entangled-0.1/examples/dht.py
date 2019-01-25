import json
import udp_common
import threading
import socket

class DHT:
    def __init__(self, boot_addr, boot_port, r_addr, r_port):
        self.boot_addr = boot_addr
        self.boot_port = boot_port
        self.r_addr = r_addr
        self.r_port = r_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        self.sock.bind((self.r_addr, self.r_port))

    def __getitem__(self, key):
        msg = {'method' : 'get', 'key' : key, 'addr' : [self.r_addr, self.r_port]}
        udp_common.send(self.boot_addr, self.boot_port, json.dumps(msg))
        data, addr = self.sock.recvfrom(1024) # buffer size is 1024 bytes
        return json.loads(data)

    def __setitem__(self, key, value):
        msg = {'method' : 'set', 'key' : key, 'value' : json.dumps(value), 'addr' : [self.r_addr, self.r_port]}
        udp_common.send(self.boot_addr, self.boot_port, json.dumps(msg))
        data, addr = self.sock.recvfrom(1024) # buffer size is 1024 bytes

    def __contains__(self, key):
        return self.__getitem__(key) != None



if __name__ == '__main__':
    print 'Start'
    dht = DHT("127.0.0.1", 6000, "127.0.0.1", 6101)
    print dht['123']
    dht['123'] = json.dumps([10])
    print dht['123']
    print dht['321']
    print '222' in dht

    #t = threading.Thread(target=udp_common.listen,  args=(handler, "127.0.0.1", 6101, lambda : False))
    #def sender():
    #    print 'Start'
    #t.run()
    #t2 = threading.Thread(target=sender, args=())
    #t2.run()
    #print 'Start'
    #print msg
