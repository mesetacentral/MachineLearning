#!/usr/bin/env bash
vscipy=$(pip show scipy)
if [ "$vscipy" == '' ];
then
  echo "scipy not installed"
  vpip=$(pip3 -V)
  if [ "$vpip" == '' ];
  then
    echo "pip3 not installed"
    apt update;
    apt install python3-pip;
  fi
  pip3 install scipy
fi

FILE="mnist.pkl"
test -f $FILE && python3 SaveMNIST.py
python3 ClientMNIST.py
