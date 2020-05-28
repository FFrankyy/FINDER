#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from FINDER import FINDER

def main():
    dqn = FINDER()
    dqn.Train()


if __name__=="__main__":
    main()
