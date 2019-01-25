import socket

def listen(handler, UDP_IP, UDP_PORT, check):
    print "in listen"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
    sock.bind((UDP_IP, UDP_PORT))
    while not check():
        try:
            data, addr = sock.recvfrom(65535) # buffer size is 1024 bytes
            handler(data, addr)
        except Exception as e:
            print 'listen err', e

def send(UDP_IP, UDP_PORT, MESSAGE):
    #print "in send"
    sock = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

