import cv2
import numpy as np
import sys

template = '''ply
format ascii 1.0           
comment made by python script
element vertex %d         
property float x        
property float y       
property float z      
property uchar red
property uchar green
property uchar blue
end_header  
'''

if __name__ == '__main__':
    name, out_name = sys.argv[1], sys.argv[2]
    img = cv2.imread(name, cv2.CV_LOAD_IMAGE_UNCHANGED)
    length = img.shape[0] * img.shape[1]
    header = template % length
    #print header
    with open(out_name, 'w') as f:
        f.write(header)
        for i in xrange(img.shape[0]):
            for j in xrange(img.shape[1]):
                x = img[i, j]
                f.write('%d %d %d %d %d %d\n' % (i, j, x[0], x[2], x[1], x[0]))    
