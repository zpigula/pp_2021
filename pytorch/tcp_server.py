import socket

def server():
    host = socket.gethostname()   # get local machine name
    port = 8080  # Make sure it's within the > 1024 $$ <65535 range
    
    s = socket.socket()
    s.bind((host, port))
    print ("Waiting for client connection ...")
    s.listen(1)
    c, addr = s.accept()
    print("Connection from: " + str(addr))
    run = True
    while run:
        data = c.recv(1024).decode('utf-8')
        if data == 'q':
            run = False  
        elif data:
            print('From online user: ' + data)
            data = data.upper()
            c.send(data.encode('utf-8'))

    c.close()

if __name__ == '__main__':
    server()