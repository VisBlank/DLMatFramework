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
        try:
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__socket.connect((self.__m_IP, self.__m_Port))
        except ConnectionRefusedError:
            print("Connection refused, check if game is ready")

    def disconnect(self):
        self.__socket.send("termina".encode())
        self.__socket.close()

    def get_image(self, pkgSize=32):
        # Send server command to ask for a image
        self.__socket.send("imagem".encode())
        # Get 4 bytes from socket
        sizeInfoBA = self.__socket.recv(4)
        # Convert bytearray(4 bytes) into int32
        sizeInfo = int.from_bytes(sizeInfoBA, byteorder='big', signed=False)   
        #print(sizeInfo)
        recBytes = 0
        dataImage = bytearray()
        while True:
            # Get some chunk of data
            data = self.__socket.recv(pkgSize)
            if not data:
                print("Nothing")
                break                        
            # Append bytearray with received data
            dataImage += data
            # Stop when received at least sizeInfo bytes
            recBytes += len(data)
            #print("Something received size %d sum_received %d" % (len(data),  recBytes))
            if recBytes >= sizeInfo:
                #print("Received enough")
                break            
        
        # Convert received byte array to PIL image
        try:
            img = Image.open(io.BytesIO(dataImage))
        except OSError:
            print("Invalid image")                        
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
        
        # Take out some unused characters
        telemetry_msg = telemetry_msg.strip("\\r\\n'")
        telemetry_msg = telemetry_msg.strip("b'")
        telemetry_msg_list = telemetry_msg.split("|")
        
        # Convert to list of floats
        telemetry_msg_list = list(map(float, telemetry_msg_list))
        
        return telemetry_msg_list
    
    def send_command(self, command):        
        # Create string on the format "motor|0.0|1.0\r\n"
        command_str = ""
        command_str += "motor|"        
        idx = 0
        for i in command:
            if idx != len(command)-1:
                command_str += str(i)+"|"        
            else:
                command_str += str(i)
            idx+=1
        
        command_str += "\r\n"

        self.__socket.send(command_str.encode())
        resp = self.__socket.recv(4)
        return resp.decode()
        