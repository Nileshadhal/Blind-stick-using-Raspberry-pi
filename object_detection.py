import cv2
import time
import pyttsx3
from PIL import Image
from threading import Lock
from pycocotools.coco import COCO
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

OUTPUT_THRESHOLD = 0.8
TEST_IMAGE_NAME = 'test_image.png'


def load_object_detection_model():
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, progress=True)


def load_coco_json_file():
    return COCO('./annotations/instances_val2017.json')


def get_threshold_passed_outputs(outputs):
    outputs[0]['boxes'] = outputs[0]['boxes'][outputs[0]['scores'] > OUTPUT_THRESHOLD]
    outputs[0]['labels'] = outputs[0]['labels'][outputs[0]['scores'] > OUTPUT_THRESHOLD]
    outputs[0]['scores'] = outputs[0]['scores'][outputs[0]['scores'] > OUTPUT_THRESHOLD]
    return outputs


def load_sample_image(image_path):
    sample_image = Image.open(image_path)
    sample_image_tensor = pil_to_tensor(sample_image)
    return sample_image_tensor.unsqueeze(dim=0) / 255.0


def main(coco_obj, object_detection_model, test_image_path):
    sample_image_tensor = load_sample_image(test_image_path)
    object_detection_model.eval()
    sample_image_predictions = object_detection_model(sample_image_tensor)
    outputs = get_threshold_passed_outputs(sample_image_predictions)
    output_labels = coco_obj.loadCats(outputs[0]['labels'].numpy())
    predicted_labels = [label['name'] for label in output_labels]
    if len(predicted_labels) > 0:
        print(f'Count Of Predicted Labels: {len(predicted_labels)}')
        print(predicted_labels)
        return list(set(predicted_labels))
    else:
        print('No Object Is Detected!')
        return None


def detect_objects_and_speak(image_frame, dist=0):
    object_str = ""
    cv2.imwrite(TEST_IMAGE_NAME, image_frame)
    predicted_objects = main(coco_obj_ref, object_detection_model_ref, TEST_IMAGE_NAME)
    if predicted_objects:
        for obj in predicted_objects:
            object_str += obj + " "
        modified_str = f'Detected Objects: {object_str} With Distance {dist} cm'
        print(modified_str)
        engine.say(modified_str)


def detect_objects_and_speak_video(vid):
    engine.startLoop(False)
    while True:
        time.sleep(4)
        ret, frame = vid.read()
        cv2.imshow('FRAME#2506', frame)
        engine.iterate()
        mutex.acquire()
        detect_objects_and_speak(frame)
        mutex.release()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            engine.endLoop()
            break


if __name__ == '__main__':
    try:
        mutex = Lock()
        engine = pyttsx3.init()
        vid_capture = cv2.VideoCapture(0)
        coco_obj_ref = load_coco_json_file()
        object_detection_model_ref = load_object_detection_model()
        time.sleep(2)
        # main(coco_obj_ref, object_detection_model_ref, './sample_images/class_staffs.jpeg')

        detect_objects_and_speak_video(vid_capture)

    except Exception as e:
        print(e)

    vid_capture.release()
    cv2.destroyAllWindows()
