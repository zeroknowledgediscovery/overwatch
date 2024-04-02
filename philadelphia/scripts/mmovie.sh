#!/bin/bash

count_=1

for i in `ls *png`
do
    count=`printf "%04d" $count_`
    cp $i 'frame_'"$count"'.png'
    count_=$((count_+1))
done


ffmpeg -pix_fmt yuv420p  \
 -s 600x800 -r 1 -start_number 1 -i frame_%04d.png \
-c:v libx264 movie.mp4



ffmpeg  -i movie.mp4  -b:v 0  -crf 30  -pass 1  -an -f webm -y /dev/null
ffmpeg  -i movie.mp4  -b:v 0  -crf 30  -pass 2  movie.webm

rm frame*
