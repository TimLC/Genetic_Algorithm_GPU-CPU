from multiprocessing import Process
import os
import datetime
from time import time

def func_1(title):
    now = datetime.datetime.now()
    print("Current microsecond: " + str(now))

def func_2(name):
    now = datetime.datetime.now()
    print("Current microsecond: " + str(now))

def start_f1(title):
    func_1(title)

def start_f2(name):
    func_2(name)

if __name__ == '__main__':
    procs = []
    procs.append(Process(target=start_f1, args = ('bob', )))
    procs.append(Process(target=start_f2, args = ('bob', )))
    map(lambda x: x.start(), procs)
    map(lambda x: x.join(), procs)