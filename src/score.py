import os
import sys

golden_file = sys.argv[1]
output_file = sys.argv[2]
s = int(sys.argv[3])
e = int(sys.argv[4])

for i in xrange(s,e+1):
    cmd = './score ../data/dic %s %s > tmp'%(golden_file,output_file + str(i))
    os.system(cmd)
    cmd = 'grep \'F MEASURE\' tmp '
    os.system(cmd)
    cmd = 'rm tmp'
    os.system(cmd)

