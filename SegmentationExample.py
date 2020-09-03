# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under NVIDIA Simple Streamer License

import torch
import numpy as np
from apex.fp16_utils import network_to_half

from StreamingTools import WebCamServer, StreamServer, Timer

def to_tensor(frames, use_cuda, use_half):

    frame = torch.from_numpy(np.array(frames))

    if use_cuda:
        frame = frame.cuda()
    
    dtype = torch.float16 if use_half else torch.float32

    frame = torch.transpose(frame, 1, 3)
    frame = torch.transpose(frame, 2, 3)
    frame = frame.type(dtype)/255
    frame[:,0,:,:] = (frame[:,0,:,:]-0.485)/0.299
    frame[:,1,:,:] = (frame[:,1,:,:]-0.456)/0.224
    frame[:,2,:,:] = (frame[:,2,:,:]-0.406)/0.225

    return frame.contiguous() 

def to_image(frame, mask, alpha=0.5):

    label = mask.argmax(0).type(torch.float32)
    R = torch.sin(label*13)**2
    G = torch.sin(label*17)**2
    B = torch.sin(label*23)**2
    label = torch.stack([R, G, B], dim=2)*255
    label = label.cpu().numpy()
    result = label*alpha+(1-alpha)*np.array(frame, dtype=np.float32)

    return result.astype(np.uint8)


if __name__ == "__main__":

    use_cuda = True                            # use CUDA or not
    use_half = True                            # use mixed precision or not
    draw_segmentation = True                   # deactivate for performance
    batch_size = 8                             # increase for performance
    
    # choose one
    model = torch.hub.load('pytorch/vision:v0.5.0', 
                           'fcn_resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.5.0', 
    #                        'deeplabv3_resnet101', pretrained=True)
    
    if use_cuda:
        model = model.cuda()

    if use_half:
        model = network_to_half(model)
    
    model.eval()

    info = False
    webcam = WebCamServer(info=info)           # this is basically your webcam
    stream = StreamServer(info=info, MBps=1.0) # this is basically your screen
    timer  = Timer()                           # optionally measure FPS
    
    with torch.no_grad():
        
        # inference loop
        print_every = max(2**6/batch_size, 1)
        for iteration in range(2**30):

            # read all frames in a batch
            batch  = [webcam.read_nowait() for _ in range(batch_size)]
            states = [entry[0] for entry in batch]
            frames = [entry[1] for entry in batch]

            # perform inference
            predict = model(to_tensor(frames, use_cuda, use_half))['out']

            # write all frames in a batch
            for success, frame, mask in zip(states, frames, predict):
                if success:
                    if draw_segmentation:
                        frame = to_image(frame, mask)
                    stream.write_nowait(frame)
        
            # exponentially moving averaged FPS 
            fps = timer.update(batch_size=batch_size)

            if iteration % print_every == print_every-1:
                print("%2.2f FPS inference loop" % fps)

