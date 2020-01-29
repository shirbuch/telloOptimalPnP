import socket
import time

drone1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
drone1.setsockopt(socket.SOL_SOCKET, 25, 'wlp4s0'.encode())
drone2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
drone2.setsockopt(socket.SOL_SOCKET, 25, 'wlxe8de27a8fe5b'.encode())

drone1.sendto('command'.encode(), 0, ('192.168.10.1', 8889))
drone2.sendto('command'.encode(), 0, ('192.168.10.1', 8889))

drone1.sendto('takeoff'.encode(), 0, ('192.168.10.1', 8889))
drone2.sendto('takeoff'.encode(), 0, ('192.168.10.1', 8889))

time.sleep(5)

drone1.sendto('command'.encode(), 0, ('192.168.10.1', 8889))
drone2.sendto('command'.encode(), 0, ('192.168.10.1', 8889))

drone1.sendto('land'.encode(), 0, ('192.168.10.1', 8889))
drone2.sendto('land'.encode(), 0, ('192.168.10.1', 8889))
