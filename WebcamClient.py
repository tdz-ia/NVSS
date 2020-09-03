# ! /usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under NVIDIA Simple Streamer License

from StreamingTools import WebCamClient

if __name__ == "__main__" :

    # start a client with dynamic with dynamic 
    # quality to ensure fixed bandwidth 1.0 MB/s
    client = WebCamClient(MBps=1.0, cam=0)
