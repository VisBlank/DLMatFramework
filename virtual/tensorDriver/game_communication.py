#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:54:33 2017

@author: leoara01
"""

import socket
import io
from PIL import Image

class GameTelemetry:
    __m_IP = '127.0.0.1'
    __m_Port = 50007
    __m_BUFFER_SIZE = 4096
    
    # Constructor
    def __init__(self, ip = '127.0.0.1', port = 50007):
        self.__m_Port = port        
        self.__m_IP = ip
    
    def connect(self):
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.connect((self.__m_IP, self.__m_Port))

    def disconnect(self):
        self.__socket.send("termina".encode())
        self.__socket.close()

    def get_image(self):
        # Send server command to ask for a image
        self.__socket.send("imagem".encode())
        # Get 4 bytes from socket
        sizeInfoBA = self.__socket.recv(4)
        # Convert bytearray(4 bytes) into int32
        sizeInfo = int.from_bytes(sizeInfoBA, byteorder='big', signed=False)   
        recBytes = 0
        dataImage = bytearray()
        while True:
            # Get some chunk of data
            data = self.__socket.recv(self.__m_BUFFER_SIZE)
            # Append bytearray with received data
            dataImage += data
            # Stop when received at least sizeInfo bytes
            recBytes += self.__m_BUFFER_SIZE
            if recBytes > sizeInfo:
                break
        
        # Convert received byte array to PIL image
        img = Image.open(io.BytesIO(dataImage))
        return img
    
    def get_game_data(self):
        # Send server command to ask for other information        
        self.__socket.send("telemetria".encode())
        # Get 4 bytes from socket
        sizeInfoBA = self.__socket.recv(4)
        # Convert bytearray(4 bytes) into int32
        sizeInfo = int.from_bytes(sizeInfoBA, byteorder='big', signed=False)
        # Get message text
        telemetry_msg = str(self.__socket.recv(sizeInfo))
        return telemetry_msg
    
    def send_command(self, command):
        #self.__socket.send("motor|0.0|1.0\r\n".encode())
        self.__socket.send(command.encode())
        resp = self.__socket.recv(4)
        return resp.decode()
        