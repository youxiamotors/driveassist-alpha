#!/bin/sh

URL_PREFIX="http://youxiamotors.com/upload/lane-detection"
DIR=$(dirname $0)

for file in 13510002.MOV 16290029.MOV 17050041.MOV
do
	wget -c $URL_PREFIX/$file -O $DIR/$file
done

exit 0
