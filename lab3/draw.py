#!/usr/bin/env python
import matplotlib.pyplot as plt 
import sys, pylab

def main():
    diag = plt.figure()
    xargs = []
    yargs = []
    i = input()
    while (i != '-1'):
        x, y = i.split(' ')
        xargs.append(int(x))
        yargs.append(int(y))
        i = input()

    pylab.plot(xargs, yargs)
    plt.savefig("out.png", fmt='png')
    return 0

if (__name__ == '__main__'):
    sys.exit(main())
