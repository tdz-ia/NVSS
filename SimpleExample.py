# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under NVIDIA Simple Streamer License

from StreamingTools import WebCamServer, StreamServer, Timer

def dummy_inference(x):
    """invert an image as cheap placeholder for inference operation"""

    return 255-x

if __name__ == "__main__":

    info = False
    webcam = WebCamServer(info=info)           # this is basically your webcam
    stream = StreamServer(info=info, MBps=1.0) # this is basically your screen
    timer  = Timer()                           # optionally measure FPS
    
    # inference loop
    print_every = 2**10
    for iteration in range(2**30):

        # this frame is read in an asynchronous fashion, i.e.
        # it is either a new one or a cached old one --
        # hence the loop is decoupled from network performance
            
        success, frame = webcam.read_nowait()
        frame = dummy_inference(frame)  
        if success:
            stream.write_nowait(frame)
        
        # exponentially moving averaged FPS 
        fps = timer.update()

        if iteration % print_every == print_every-1:
            print("%2.2f FPS inference loop" % fps)
