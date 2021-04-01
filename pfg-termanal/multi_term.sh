#!/bin/bash

for file in $1/*.pdb
do
  python3 pfg_run.py --p $file --o $2
  if [ -f termanal_support ]; then
    rm termanal_support
  fi
done
