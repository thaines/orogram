#!/bin/bash

rm -R build
rm -R dist
rm -R Orogram.egg-info

cd design
rm *.pdf
cd ..

cd test
rm *.pdf
rm *.svg
cd ..

cd orogram
rm -R __pycache__
rm *.c
cd ..
