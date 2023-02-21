import selectors
import socket
import sys
import os
import fcntl

m_selector = selectors.DefaultSelector()

# set sys.stdin non-blocking
def set_input_nonblocking():
    orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)

def create_socket(port, max_conn):
    host = socket.gethostname()  # get local machine name
    server_addr = (host, port)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setblocking(False)
    server.bind(server_addr)
    server.listen(max_conn)
    return server

def read(conn, mask):
    global GO_ON
    client_address = conn.getpeername()
    data = conn.recv(1024)
    print('Got {} from {}'.format(data, client_address))
    if not data:
         GO_ON = False

def accept(sock, mask):
    new_conn, addr = sock.accept()
    new_conn.setblocking(False)
    print('Accepting connection from {}'.format(addr))
    m_selector.register(new_conn, selectors.EVENT_READ, read)

def quit():
    global GO_ON
    print('Exiting...')
    GO_ON = False


def from_keyboard(arg1, arg2):
    line = arg1.read()
    if line == 'quit\n':
        quit()
    else:
        print('User input: {}'.format(line))

GO_ON = True
set_input_nonblocking()

# listen to port 10000, at most 10 connections
server = create_socket(8080, 10)

m_selector.register(server, selectors.EVENT_READ, accept)
m_selector.register(sys.stdin, selectors.EVENT_READ, from_keyboard)

while GO_ON:
    #sys.stdout.write('>>> ')
    #sys.stdout.flush()
    for k, mask in m_selector.select(-1):
        callback = k.data
        callback(k.fileobj, mask)


# unregister events
m_selector.unregister(sys.stdin)

# close connection
server.shutdown(socket.SHUT_RDWR)
server.close()

#  close select
m_selector.close()