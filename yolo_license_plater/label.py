import cv2
import re
import os

imagine_path = 'D:\\programm\\machine_learning\\deep_learning\\YOLO\\yolov5-face\\dataSet\\images\\val'
label_path = 'D:\\programm\\machine_learning\\deep_learning\\YOLO\\yolov5-face\\dataSet\\labels\\val'

def re_label(filename):
    l = filename.split('-')
    print(l[2])
    box_coordinate = re.match(r'(\d*)[&](\d*)[_](\d*)[&](\d*)',l[2])
    box_coordinate=[box_coordinate.group(i) for i in range(1,5)]
    box_coordinate[0] = int(box_coordinate[0])-10
    box_coordinate[1] = int(box_coordinate[1])-10
    box_coordinate[2] = int(box_coordinate[2])+10
    box_coordinate[3] = int(box_coordinate[3])+10
    vertex_coordinate = re.match(r'(\d*)[&](\d*)[_](\d*)[&](\d*)[_](\d*)[&](\d*)[_](\d*)[&](\d*)',l[3])
    vertex_coordinate=[int(vertex_coordinate.group(i)) for i in range(1,9)]
    return box_coordinate, vertex_coordinate


def label_visualization(filename):
    i=cv2.imread(os.path.join(imagine_path,filename))
    box_coordinate, vertex_coordinate = re_label(filename)
    cv2.rectangle(i,(box_coordinate[0],box_coordinate[1]),(box_coordinate[2],box_coordinate[3]) , (0,255,0),2)
    cv2.circle(i,(vertex_coordinate[0],vertex_coordinate[1]),2,(0,0,255),2)
    cv2.circle(i,(vertex_coordinate[2],vertex_coordinate[3]),2,(0,0,255),2)
    cv2.circle(i,(vertex_coordinate[4],vertex_coordinate[5]),2,(0,0,255),2)
    cv2.circle(i,(vertex_coordinate[6],vertex_coordinate[7]),2,(0,0,255),2)
    cv2.imshow('image',i)
    cv2.waitKey(0)

def write_label(filename):
    box_coordinate, vertex_coordinate = re_label(filename)
    with open(os.path.join(label_path,filename.replace('.jpg','.txt')),'w') as f:
        f.write('0 ')
        f.write(str((box_coordinate[0]+box_coordinate[2])/2/720)+' ')
        f.write(str((box_coordinate[1]+box_coordinate[3])/2/1160)+' ')
        f.write(str((box_coordinate[2]-box_coordinate[0])/720)+' ')
        f.write(str((box_coordinate[3]-box_coordinate[1])/1160)+' ')
        for i in range(4):
            f.write(str(vertex_coordinate[2*i]/720)+' ')
            f.write(str(vertex_coordinate[2*i+1]/1160)+' ')
            
        f.write(str(vertex_coordinate[0]/720)+' ')
        f.write(str(vertex_coordinate[2*i+1]/1160)+' ')
        f.write('\n')
    return

if __name__ == '__main__':
    for filename in os.listdir(imagine_path):
        write_label(filename)