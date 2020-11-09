#!/bin/bash
fileid="1seVcZ9TE0ssrQHQdfVveR8rjH-_x0R-t"
filename="tiny_imagenet.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

unzip tiny_imagenet.zip