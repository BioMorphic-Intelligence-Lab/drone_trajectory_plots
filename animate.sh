#!/bin/sh
ffmpeg -framerate 100 -i frames/frame_%05d.png -c:v libx264 -r 30 -pix_fmt yuv420p animations/many_3d_anim.mp4