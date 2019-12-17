# Boolean fn: ABCDE -> A¬C¬E
#          16  8  4  2  1
import random

dataset = ["0, 0, 0, 0, 0",     "0, 1, 1",
           "0, 0, 0, 0, 1",     "0, 1, 0",
           "0, 0, 0, 1. 0",     "0, 1, 1",
           "0, 0, 0, 1, 1",     "0, 1, 0",
           "0, 0, 1, 0, 0",     "0, 0, 1",
           "0, 0, 1, 0, 1",     "0, 0, 0",
           "0, 0, 1, 1, 0",     "0, 0, 1",  #t
           "0, 0, 1, 1, 1",     "0, 0, 0",
           "0, 1, 0, 0, 0",     "0, 1, 1",
           "0, 1, 0, 0, 1",     "0, 1, 0",
           "0, 1, 0, 1, 0",     "0, 1, 1",  #t
           "0, 1, 0, 1, 1",     "0, 1, 0",
           "0, 1, 1, 0, 0",     "0, 0, 1",
           "0, 1, 1, 0, 1",     "0, 0, 0",
           "0, 1, 1, 1, 0",     "0, 0, 1",
           "0, 1, 1, 1, 1",     "0, 0, 0",  #t
           "1, 0, 0, 0, 0",     "1, 1, 1",
           "1, 0, 0, 0, 1",     "1, 1, 0",
           "1, 0, 0, 1, 0",     "1, 1, 1",  #t
           "1, 0, 0, 1, 1",     "1, 1, 0",
           "1, 0, 1, 0, 0",     "1, 0, 1",
           "1, 0, 1, 0, 1",     "1, 0, 0",
           "1, 0, 1, 1, 0",     "1, 0, 1",
           "1, 0, 1, 1, 1",     "1, 0, 0",
           "1, 1, 0, 0, 0",     "1, 1, 1",
           "1, 1, 0, 0, 1",     "1, 1, 0",
           "1, 1, 0, 1, 0",     "1, 1, 1",  #t
           "1, 1, 0, 1, 1",     "1, 1, 0",
           "1, 1, 1, 0, 0",     "1, 0, 1",
           "1, 1, 1, 0, 1",     "1, 0, 0",
           "1, 1, 1, 1, 0",     "1, 0, 1",  #t
           "1, 1, 1, 1, 1",     "1, 0, 0"
]

set1 = set()
#randomly choose 6 for testing - then mark above
while(len(set1) < 6):
    pos = random.randrange(0, 62, 2)
    set1.add(dataset[pos])
    print(set1)
   # print(dataset[pos] + "\t" + dataset[pos+1])

testing = ['0, 1, 1, 1, 1',     '0, 0, 0',
           '1, 1, 0, 1, 0',     '1, 1, 1',
           '0, 0, 1, 1, 0',     '0, 0, 1',
           '1, 1, 1, 1, 0',     '1, 0, 1',
           '1, 0, 0, 1, 0',     '1, 1, 1',
           '0, 1, 0, 1, 0',     '0, 1, 1'
]

training = ["0, 0, 0, 0, 0",     "0, 1, 1",
            "0, 0, 0, 0, 1",     "0, 1, 0",
            "0, 0, 0, 1. 0",     "0, 1, 1",
            "0, 0, 0, 1, 1",     "0, 1, 0",
            "0, 0, 1, 0, 0",     "0, 0, 1",
            "0, 0, 1, 0, 1",     "0, 0, 0",
            "0, 0, 1, 1, 1",     "0, 0, 0",
            "0, 1, 0, 0, 0",     "0, 1, 1",
            "0, 1, 0, 0, 1",     "0, 1, 0",
            "0, 1, 0, 1, 1",     "0, 1, 0",
            "0, 1, 1, 0, 0",     "0, 0, 1",
            "0, 1, 1, 0, 1",     "0, 0, 0",
            "0, 1, 1, 1, 0",     "0, 0, 1",
            "1, 0, 0, 0, 0",     "1, 1, 1",
            "1, 0, 0, 0, 1",     "1, 1, 0",
            "1, 0, 0, 1, 1",     "1, 1, 0",
            "1, 0, 1, 0, 0",     "1, 0, 1",
            "1, 0, 1, 0, 1",     "1, 0, 0",
            "1, 0, 1, 1, 0",     "1, 0, 1",
            "1, 0, 1, 1, 1",     "1, 0, 0",
            "1, 1, 0, 0, 0",     "1, 1, 1",
            "1, 1, 0, 0, 1",     "1, 1, 0",
            "1, 1, 0, 1, 1",     "1, 1, 0",
            "1, 1, 1, 0, 0",     "1, 0, 1",
            "1, 1, 1, 0, 1",     "1, 0, 0",
            "1, 1, 1, 1, 1",     "1, 0, 0"
]