# NVIDIA Simple Streamer (NVSS)

## About
NVSS is a simple yet powerful collection of convenient Python classes for streaming video from and to deep learning applications using a server-client architecture. This allows for decoupling of the webcam and display device from the inference solution running either locally, remotely in a cloud instance, a container, or a combination thereof.

### Transparent Performance - Awesome Demo Capabilities
Furthermore, NVSS employs non-blocking queues for both the incoming webcam stream and the outgoing video stream to decouple inference performance from the fixed frame rate of the webcam and bandwidth constraints imposed by the network connection. As a result, NVSS is ideally suited to demonstrate the impact of  optimization approaches, e.g. TensorRT, while still allowing to visualize the result. NVSS supports both dynamic adjustment of image quality respecting network constraints or fixed compression quality for best visual results. Finally, NVSS allows for forwarding video streams using simple network ports. Hence, you can map a webcam from anywhere directly into a container running in the cloud.

### Runs Everywhere
NVSS is based on Python's standard library, OpenCV, and Numpy. That's it. These dependencies can easily be met on Linux and Windows systems. Python 2, however, is not supported and never will be.

## Quickstart

First, make sure you have a working Python 3 environment with Numpy and OpenCV. Subsequently, start the simple example which (i) reads images from a client's webcam, (ii) subsequently inverts them on the server, and (iii) finally streams them back to the same or another client of your choice.
```
username@server:~/NVSS$ python3 SimpleExample.py 
webcam server: waiting for connections
stream server: waiting for connections 
```
Afterwards, start the stream client on the same machine or any other machine which can access port 8090 of the server:
```
username@client:~/NVSS$ python3 StreamClient.py 
```
The server should now be aware of the stream client:

```
username@server:/NVSS$ python3 SimpleExample.py 
webcam server: waiting for connections
stream server: waiting for connections 
stream server: listening to client ('127.0.0.1', 49978)
```
Start the webcam client on the same machine or any other machine which can access port 8089 of the server and has a working webcam attached to it:
```
username@client:~/NVSS$ python3 WebcamClient.py 
30.0
webcam client sending data: 0.09/30.00 FPS 	 0.01/1.00 MiB/s 	 95 Quality
webcam client sending data: 3.97/30.00 FPS 	 0.30/1.00 MiB/s 	 96 Quality
webcam client sending data: 7.52/30.00 FPS 	 0.61/1.00 MiB/s 	 97 Quality
...
```
The server is now aware of the second client and starts inference:

```
username@server:~/NVSS$ python3 SimpleExample.py 
webcam server: waiting for connections
stream server: waiting for connections 
stream server: listening to client ('127.0.0.1', 49978)
webcam server: listening to client ('127.0.0.1', 38292)
7178.38 FPS inference loop
7519.14 FPS inference loop
7524.10 FPS inference loop
7350.81 FPS inference loop
7339.40 FPS inference loop
...
```
You should now see yourself in inverted colors. Obviously, the performance of the inference loop with approximately 7000 FPS is decoupled from the 30 FPS of the webcam.

## Container Usage and NVIDIA GPUs

Launch an interactive Pytorch docker container and forward the ports 8089 as well as 8090 from the baremetal machine (host) into the container:
```
username@host:~/NVSS$ docker run --gpus all -it --rm -p 8089:8089 -p 8090:8090 -v /home/username/NVSS:/workspace/NVSS nvcr.io/nvidia/pytorch:20.03-py3
```
Inside the container launch a complex example performing classification or semantic segmentation:
```
root@container:/workspace/NVSS# python3 SegmentationExample.py 
webcam server: waiting for connections
stream server: waiting for connections 
```
Finally launch the clients outside of the container on the baremetal machine as described above or forward the ports 8089 and 8090 using OpenSHH from your laptop.

# NVSS
# NVSS:https://developer.nvidia.com/blog/streaming-interactive-deep-learning-applications-at-peak-performance/#comments 
