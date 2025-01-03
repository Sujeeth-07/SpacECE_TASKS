import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def load_yolo_model(config_path, weights_path, classes_path):
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return net, classes, output_layers
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None, None, None

def detect_objects(net, output_layers, frame, classes, confidence_threshold=0.5):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    safe_objects = ["pen", "pencil", "toy", "book", "cup", "chair", "bed"]
    harmful_objects = ["knife", "scissors"]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    if indexes is None:
        return frame, []
    results = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if label in safe_objects else (0, 0, 255) if label in harmful_objects else (255, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if label in safe_objects:
                results.append(5)  # Positive
            elif label in harmful_objects:
                results.append(-1)  # Negative
            else:
                results.append(2)  # Neutral
    return frame, results

def calculate_room_brightness(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_image)

def get_dominant_color(image):
    resized_image = cv2.resize(image, (150, 150))
    data = resized_image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(data)
    return tuple(map(int, kmeans.cluster_centers_[0]))

def process_image(image_path, net, classes, output_layers, output_folder):
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None
    except Exception as e:
        print(f"Error reading image: {e}")
        return None, None

    processed_frame, results = detect_objects(net, output_layers, frame.copy(), classes)
    brightness = calculate_room_brightness(frame)
    dominant_color = get_dominant_color(frame)
    color_info = f"Brightness: {brightness:.2f}, Dominant Color: {dominant_color}"
    cv2.putText(processed_frame, color_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save output
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_folder, f"{name}_processed{ext}")
    cv2.imwrite(output_path, processed_frame)
    return results, output_path

def process_folder(input_folder, net, classes, output_layers, output_folder):
    if not os.path.exists(input_folder) or not os.path.isdir(input_folder):
        print(f"Error: Invalid folder path: {input_folder}")
        return

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))]
    if not image_files:
        print(f"No images found in folder: {input_folder}")
        return

    all_results = {}
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        print(f"Processing image: {image_path}")
        results, output_path = process_image(image_path, net, classes, output_layers, output_folder)
        if results is not None:
            all_results[output_path] = results

    if all_results:
        for output_path, results in all_results.items():
            if all(x == 5 for x in results):  # All results are Positive
                print(f"{output_path}: Overall result: Positive (Score: 5)")
            elif all(x == -1 for x in results):  # All results are Negative
                print(f"{output_path}: Overall result: Negative (Score: -1)")
            elif any(x == -1 for x in results):  # Some results are Negative
                print(f"{output_path}: Overall result: Mixed (Contains Negative) (Score: -1)")
            else:  # Neutral result
                print(f"{output_path}: Overall result: Neutral (Score: 2)")
    else:
        print("No images were processed to perform overall analysis.")


if __name__ == "__main__":
    base_folder = r"D:\Obj"  # your main folder
    image_folder = os.path.join(base_folder, "images")  # Construct path to 'images' folder
    config_path = os.path.join(base_folder, "yolov3.cfg")
    weights_path = os.path.join(base_folder, "yolov3.weights")
    classes_path = os.path.join(base_folder, "coco.names")
    output_folder = os.path.join(base_folder, "detected_images")

    # Check if files exist
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        exit()
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found: {weights_path}")
        exit()
    if not os.path.exists(classes_path):
        print(f"Error: Classes file not found: {classes_path}")
        exit()
    if not os.path.exists(image_folder):  # check for image folder existence
        print(f"Error: Image folder not found: {image_folder}")
        exit()

    net, classes, output_layers = load_yolo_model(config_path, weights_path, classes_path)
    if net is None or classes is None or output_layers is None:  # check if model loading is successful
        print("Error: Model loading failed.")
        exit()

    process_folder(image_folder, net, classes, output_layers, output_folder)
