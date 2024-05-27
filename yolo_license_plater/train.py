from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')

    result = model.train(data='.\\yolo_license_plater\\dataSet\\LP.yaml',device='cpu')