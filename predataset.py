import os
import cv2
from LP_detect.detect_fix import detect_LP


if __name__ == '__main__':
    filepath=r'./LP_detect/dataset/raw_imagine'
    savepath=r'./LP_detect/dataset/box_imagine'

    try:
        os.makedirs(savepath)
        print('Directory created')
    except:
        pass

    for file in os.listdir(filepath):
        try:
            process=detect_LP(os.path.join(filepath,file))
            cv2.imwrite(os.path.join(savepath,file),process)
        except:
            print('Error: ',file)
            pass