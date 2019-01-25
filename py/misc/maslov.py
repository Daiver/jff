import os
import sys

#maketx -v -u —oiio —checknan —filter lanczos3 D:/Projects/Skazka/Sourceimages/Layout/Roads_UDIM/Asphalt_1001.tif -o D:/Projects/Skazka/Sourceimages/Layout/Roads_UDIM/Asphalt_1001.tx
if __name__ == '__main__':
    commandWithFlags = 'maketx -v -u —oiio —checknan —filter lanczos3 '
    #dirWithTextures = 'D:/Projects/Skazka/Sourceimages/Layout/Roads_UDIM/'
    dirWithTextures = 'data/'
    targetNames = [x[-4:] in [".tif"] for x in os.listdir(dirWithTextures)]
    print targetNames
