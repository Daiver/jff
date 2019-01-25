import os
import sys

rootName = '/tmp/'
if len(sys.argv) > 1:
    rootName = sys.argv[1]

for root, directories, filenames in os.walk(rootName):
    for directory in directories:
            print 'dir', os.path.join(root, directory) 
    for filename in filenames: 
            print 'file', os.path.join(root,filename)
