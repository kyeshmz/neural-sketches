#!/bin/bash

rsync -av ./ fabai:~/gyre/ &
jupyter notebook &
ssh -NL 8000:localhost:8000 fabai &