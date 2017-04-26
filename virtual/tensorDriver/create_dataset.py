#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:06:12 2017

@author: leoara01
"""
import argparse
import game_communication
import threading
import queue

from matplotlib.pyplot import imshow
from matplotlib.pyplot import imsave
import matplotlib.pyplot as plt

class GameRecord:
    __m_id = 0
    __m_img = 0
    __m_telemetry = []
    
    def __init__(self, id_record, img, telemetry):
        self.__m_id = id_record
        self.__m_img = img 
        self.__m_telemetry = telemetry
    
    def get_id(self):
        return self.__m_id
    
    def get_image(self):
        return self.__m_img    
    
    def get_telemetry(self):
        return self.__m_telemetry

# Parser command arguments
# Reference:
# https://www.youtube.com/watch?v=cdblJqEUDNo
parser = argparse.ArgumentParser(description='Create driving dataset from game server')
parser.add_argument('--ip', type=str, required=False, default='127.0.0.1', help='Server IP address')
parser.add_argument('--port', type=int, required=False, default=50007, help='Server TCP/IP port')
parser.add_argument('-o','--outdir', type=str, required=True ,help='Server TCP/IP port')
args = parser.parse_args()

# https://www.tutorialspoint.com/python3/python_multithreading.htm
list_game_state = queue.Queue(20)

# Consumer thread that will get the states and save on disk
# Reference:
# http://www.bogotobogo.com/python/Multithread/python_multithreading_Synchronization_Producer_Consumer_using_Queue.php
class ConsumerThread(threading.Thread)    :
    __m_outdir = "./"
    
    # Constructor
    def __init__(self, outDir="./", group=None, target=None, name=None,args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        self.__m_outdir = outDir        
        return

    def run(self):
        while True:
            # Start if there is something to do
            if not list_game_state.empty():
                # Get list of items
                list_items = list_game_state.get()
                # Iterate on list saving elements to disk
                for i in list_items:
                    imsave(self.__m_outdir + "/" + str(i.get_id()) + ".png", i.get_image())
        return    


def connect_and_create_dataset(ip, port):    
    print(ip)
    print(port)
        
    comm = game_communication.GameTelemetry(ip,port)
    comm.connect()
    
    # Run until Crtl-C
    img_index = 0    
    try:
        list_records = []
        while True:
            # Get telemetry and image
            telemetry = comm.get_game_data()
            cam_img = comm.get_image()
            
            # Skip entire record if image is invalid
            if cam_img is None:
                continue;
                
            img_index += 1
            
            # Add elements on the list until it's bigger than 20 elements
            if len(list_records) < 20:
                list_records.append(GameRecord(img_index, cam_img, telemetry))
            else:
                # From the main thread add item on the queue
                if not list_game_state.full():
                    print("Add elements to assync queue")
                    list_game_state.put(list_records)
                    
                list_records = []
                list_records.append(GameRecord(img_index, cam_img, telemetry))                                
            
            print(telemetry)
    except KeyboardInterrupt:
        pass    
    
# Python main    
if __name__ == "__main__":
    # Create consumer thread (only execute when there is something to do)
    consume_game_states = ConsumerThread(name='consumer', outDir=args.outdir)    
    consume_game_states.start()
    connect_and_create_dataset(args.ip, args.port)    