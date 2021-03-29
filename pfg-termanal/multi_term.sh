#!/bin/bash

for file in $1/*.pdb
do
  python3 pfg_run.py --p $file --o $2
  rm termanal_support
done
