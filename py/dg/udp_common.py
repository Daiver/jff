import socket

def listen(handler, UDP_IP, UDP_PORT, check):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
    sock.bind((UDP_IP, UDP_PORT))
    while not check():
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        handler(data, addr)

def send(UDP_IP, UDP_PORT, MESSAGE):
    #print "in send"
    sock = socket.socket(socket.AF_INET, # Internet
                                  socket.SOCK_DGRAM) # UDP
    sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))

