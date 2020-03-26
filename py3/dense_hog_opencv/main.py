import cv2


img = cv2.imread("/home/daiver/Frame00000.jpg")


cell_size = (16, 16)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 9  # number of orientation bins

# winSize is the size of the image cropped to an multiple of the cell size
# cell_size is the size of the cells of the img patch over which to calculate the histograms
# block_size is the number of cells which fit in the patch
hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

desc = hog.compute(img)
# desc = hog.computeGradient(img)

print(desc.shape)
print(img.shape)

cv2.imshow("img", img)
cv2.waitKey()
