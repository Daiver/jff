import numpy as np
import cv2
import csv
import urllib2
import os

if __name__ == '__main__':

    # fname    = '/home/daiver/coding/data/lpwf/kbvt_lfpw_v1_train.csv'
    # dest_dir = '/home/daiver/coding/data/lpwf/train/'
    fname    = '/home/daiver/coding/data/lpwf/kbvt_lfpw_v1_test.csv'
    dest_dir = '/home/daiver/coding/data/lpwf/test/'
    
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        reader.next()
        for index, row in enumerate(reader):
            if row[1] == 'average':
                print index, row[0]
                n_attempts = 0
                n_max_attempts = 2
                while n_attempts < n_max_attempts:
                    try:
                        response = urllib2.urlopen(row[0], timeout = 10)
                        img = response.read()
                        f = open(os.path.join(dest_dir, '%d.jpg' % index), 'wb')
                        f.write(img)
                        f.close()
                        print 'OK', len(img)
                        break
                    except Exception as e:
                        print n_attempts, e
                        n_attempts += 1

