# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under NVIDIA Simple Streamer License

import multiprocessing as mp
import numpy as np
import time
import socket
import cv2

import sys
if sys.version_info[0] < 3:
    raise Exception("Only Python 3 supported")

def adjust(quality, bandwidth, rate, lower, upper):

    if bandwidth < rate:
        return min(quality+1, upper)
    else:
        return max(quality-1, lower)

class Timer:

    def __init__(self, alpha=0.1, epsilon=1E-6):
        self.fps = 0.0
        self.bnw = 0.0
        self.alpha = alpha
        self.ying = time.time()
        self.epsilon = epsilon

    def update(self, batch_size=1, bandwidth=None):

        # compute inverse time interval
        self.yang = time.time()
        self.rtau = batch_size/max(self.yang-self.ying, self.epsilon)
        self.ying = self.yang

        # update fps and bandwidth estimate
        self.fps += self.alpha*(self.rtau-self.fps)
        if bandwidth:
            self.bnw  += self.alpha*(bandwidth*self.rtau-self.bnw)
       
        return (self.fps, self.bnw) if bandwidth else self.fps

class WebCamServer:

    def __init__(self, host=None, port=None, info=None, maxQ=None, chnk=None, ncon=None):

        self.host = host if host else ''
        self.port = port if port else 8089
        self.ncon = ncon if ncon else 1
        self.chnk = chnk if chnk else 2**12
        self.maxQ = maxQ if maxQ else 16
        self.info = info if info else False
        
        self.skip = 8
        self.queue = mp.Queue(self.maxQ)
        self.cache = None
        self.worker = mp.Process(target=self.__listen__, args=(self.queue,))
        self.worker.start()

    def __listen__(self, queue):

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self.ncon)

            print("webcam server: waiting for connections")

            client, address = server.accept()

            print("webcam server: listening to client %s" % str(address))

            data = b''
            timer = Timer()

            while True:

                while len(data) < self.skip:
                    data += client.recv(self.chnk)

                packed_msg_size = data[:self.skip]
                data = data[self.skip:]
                msg_size = np.frombuffer(packed_msg_size, dtype=">u8")[0]

                while len(data) < msg_size:
                    data += client.recv(self.chnk)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                try:                
                    frame = np.fromstring(frame_data, np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    if not queue.full():
                        queue.put_nowait(frame)
                    elif self.info:
                        print("webcam server: skipped frame (queue full)")
                except:
                    if self.info:
                        print("webcam server: skipped frame (jpeg decode error)")
                    continue
                
                # estimate moving frame rate and bandwidth statistics
                if self.info:
                    fps, bw = timer.update(bandwidth=len(frame_data))

                    # useful stats
                    if self.info:
                        print("webcam server reading data: %2.2f FPS \t  %2.2f MiB/s" % (fps, bw/2**20))

    def __del__(self):
        self.worker.terminate()

    def read_nowait(self):

        try:
            success = True
            self.cache = self.queue.get_nowait()
        except:
            success = False            
            self.cache = self.queue.get() if self.cache is None else self.cache

        return success, self.cache

    def read_wait(self):

        return True, self.queue.get()

class StreamServer:

    def __init__(self, host=None, port=None, info=None, maxQ=None, ncon=None, qual=None, MBps=None):

        self.host = host if host else ''
        self.port = port if port else 8090
        self.ncon = ncon if ncon else 1
        self.maxQ = maxQ if maxQ else 16
        self.info = info if info else False
        self.qual = qual if qual else 95
        self.rate = MBps*2**20 if MBps else float("infinity")

        self.cache = None
        self.queue = mp.Queue(self.maxQ)
        self.worker = mp.Process(target=self.__listen__, args=(self.queue,))
        self.worker.start()


    def __listen__(self, queue):
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:

            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(self.ncon)

            print("stream server: waiting for connections ")

            client, address = server.accept()

            print("stream server: listening to client %s" % str(address))
          
            timer = Timer()

            while True:

                # read a frame from the queue
                frame = self.queue.get()

                # encode to jpeg with potentially dynamic quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.qual]
                success, a_numpy = cv2.imencode('.jpg', frame, encode_param)
                data = a_numpy.tostring()

                # send the jpeg frame over the network
                message_size = np.array([len(data)], dtype=">u8").tostring()

                try:
                    client.sendall(message_size + data)
                except:
                    print("stream server: frame skipped (send failed)")
                    continue
                
            
                # estimate moving frame rate and bandwidth statistics
                if self.info or self.rate:
                    fps, bw = timer.update(bandwidth=len(data))

                    # useful stats
                    if self.info:
                        print("stream server sending data: %2.2f FPS \t \
                               %2.2f/%2.2f MiB/s \t %d Quality" % 
                             (fps, bw/2**20, self.rate/2**20, self.qual)  )

                    # adjust jpeg encoding quality if fixed bandwidth specified
                    if self.rate:
                        self.qual = adjust(self.qual, bw, self.rate, 10, 100)
                        
    def __del__(self):
        self.worker.terminate()
        
    def write_nowait(self, frame):

        try:                   
            self.queue.put_nowait(frame)
        except:
            if self.info:
                print("stream server: frame skipped (queue full)")
        
    def write_wait(self, frame):
        self.queue.put(frame)

class StreamClient :

    def __init__(self, host=None, port=None, info=None, chnk=None):

        self.host = host if host else "localhost"
        self.port = port if port else  8090
        self.info = info if info else  True
        self.chnk = chnk if chnk else 2**12
        self.skip = 8
        
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as client:
        
            client.connect((self.host, self.port))
        
            data = b''

            # frame rate and bandwidth estimates, decay rate for EMA and time
            timer = Timer()

            while True:

                while len(data) < self.skip:
                    data += client.recv(self.chnk)

                packed_msg_size = data[:self.skip]
                data = data[self.skip:]
                msg_size = np.frombuffer(packed_msg_size, dtype=">u8")[0]

                while len(data) < msg_size:
                    data += client.recv(self.chnk)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                try:                
                    frame = np.fromstring(frame_data, np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
                except:
                    if self.info:
                        print("stream client: skipped frame (jpeg decode error)")
                    continue

                # estimate moving frame rate and bandwidth statistics
                if self.info or self.rate:
                    fps, bw = timer.update(bandwidth=len(frame_data))

                    # useful stats
                    if self.info:
                        print("stream client receiving data: %2.2f FPS \t %2.2f MiB/s" % (fps, bw/2**20))

class WebCamClient :

    def __init__(self, host=None, port=None, cam=None, qual=None, info=None, MBps=None, wcfg=None):

        self.host = host if host else "localhost"
        self.port = port if port else  8089
        self.cam  = cam  if cam  else  0
        self.qual = qual if qual else  95
        self.info = info if info else  True
        self.wcfg = wcfg if wcfg else (640, 480, 30)
        self.rate = MBps*2**20 if MBps else float("infinity")

        self.capture = cv2.VideoCapture(self.cam)
        
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.wcfg[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,self.wcfg[1])
        self.capture.set(cv2.CAP_PROP_FPS,         self.wcfg[2])
        self.mfps = self.capture.get(cv2.CAP_PROP_FPS)
      
        print(self.capture.get(cv2.CAP_PROP_FPS))

        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as client:
            client.connect((self.host, self.port))

            # frame rate and bandwidth estimates, decay rate for EMA and time
            timer = Timer()

            while True:

                # read a frame from the webcam
                success, frame = self.capture.read()

                # encode to jpeg with potentially dynamic quality
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.qual]
                success, a_numpy = cv2.imencode('.jpg', frame, encode_param)
                data = a_numpy.tostring()

                # send the jpeg frame over the network
                message_size = np.array([len(data)], dtype=">u8").tostring()

                try:
                    client.sendall(message_size + data)
                except:
                    if self.info:
                        print("webcam client: frame skipped (send failed)")
                    continue

                # estimate moving frame rate and bandwidth statistics
                if self.info or self.rate:
                    fps, bw = timer.update(bandwidth=len(data))

                    # useful stats
                    if self.info:
                        print("webcam client sending data: %2.2f/%2.2f FPS \t %2.2f/%2.2f MiB/s \t %d Quality" % 
                         (fps, self.mfps, bw/2**20, self.rate/2**20, self.qual))

                    # adjust jpeg encoding quality if fixed bandwidth specified
                    if self.rate:
                        self.qual = adjust(self.qual, bw, self.rate, 10, 100)