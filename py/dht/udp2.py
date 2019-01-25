import threading
from udp_common import send, listen

def simple_echo(data, addr):
    print data, addr

e1 = threading.Event()
t1 = threading.Thread(target=listen, args=(simple_echo, "127.0.0.1", 5000, e1.isSet))
t1.start()

import time
send("127.0.0.1", 5000, "Hi")
time.sleep(0.5)
send("127.0.0.1", 5000, "H")
time.sleep(0.5)
send("127.0.0.1", 5000, "Hi")
send("127.0.0.1", 5000, "Hi")

e1.set()
t1.join()
