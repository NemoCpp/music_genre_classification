#!/bin/bash

for audio in *au; do
	len=${#audio}
	name=${audio:0:len-3}
	ext=\.wav
	name=$name$ext
	sox $audio ./data/$(basename $name)
done
