#!/bin/sh

URL_PREFIX="http://youxiamotors.com/upload/lane-detection"

wget -c $URL_PREFIX/13510002.MOV
wget -c $URL_PREFIX/16290029.MOV
wget -c $URL_PREFIX/17050041.MOV

exit 0
