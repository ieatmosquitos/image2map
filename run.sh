#!/bin/bash
files=`ls images`;
cd build;
for f in $files
	do
	./MapCreator ../images/$f;
done
