#!/bin/sh
ffmpeg -framerate 25 -i frames/frame_%05d.png -c:v libx264 -r 25 -pix_fmt yuv420p animations/many_3d_anim.mp4
