
import os
import cv2
import matplotlib.pyplot as plt
import easyocr
import numpy as np
import json

def merge_english_boxes(boxes, texts, probs, max_distance=30):
    merged_boxes = []
    merged_texts = []
    merged_probs = []

    for i, (box, text, prob) in enumerate(zip(boxes, texts, probs)):
        if text.isascii():  
            if i == 0:
                merged_boxes.append(box)
                merged_texts.append(text)
                merged_probs.append(prob)
            else:
                prev_box = merged_boxes[-1]
                if abs(box[0][0] - prev_box[1][0]) < max_distance and abs(box[0][1] - prev_box[1][1]) < max_distance:
                    merged_boxes[-1] = (
                        (min(prev_box[0][0], box[0][0]), min(prev_box[0][1], box[0][1])),
                        (max(prev_box[1][0], box[1][0]), max(prev_box[1][1], box[1][1])),
                        (max(prev_box[2][0], box[2][0]), max(prev_box[2][1], box[2][1])),
                        (min(prev_box[3][0], box[3][0]), min(prev_box[3][1], box[3][1]))
                    )
                    merged_texts[-1] += ' ' + text
                    merged_probs[-1] = (merged_probs[-1] + prob) / 2  
                else:
                    merged_boxes.append(box)
                    merged_texts.append(text)
                    merged_probs.append(prob)
    
    return merged_boxes, merged_texts, merged_probs

def blur_region(image, tl, br, blur_intensity=31):
    h, w = image.shape[:2]
    tl = (max(0, tl[0]), max(0, tl[1]))
    br = (min(w, br[0]), min(h, br[1]))

    if tl[0] < br[0] and tl[1] < br[1]:
        roi = image[tl[1]:br[1], tl[0]:br[0]]
        blur_intensity = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
        blurred_roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 0)
        image[tl[1]:br[1], tl[0]:br[0]] = blurred_roi

def process_images_in_directory(directory_path, confidence_threshold=0.5, blur_intensity=31, output_dir="output"):
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                results = reader.readtext(image)
                boxes, texts, probs = zip(*results)
                merged_boxes, merged_texts, merged_probs = merge_english_boxes(boxes, texts, probs)
                merged_results = list(zip(merged_boxes, merged_texts, merged_probs))

                image_detected = image.copy()
                image_blurred = image.copy()

                output_data = {
                    "image_id": filename,
                    "english_text": [],
                    "english_text_coordinates": []
                }

                for (bbox, text, prob) in merged_results:
                    if text.isascii() and prob > confidence_threshold:
                        (top_left, top_right, bottom_right, bottom_left) = bbox
                        top_left = tuple(map(int, top_left))
                        bottom_right = tuple(map(int, bottom_right))

                        cv2.rectangle(image_detected, top_left, bottom_right, (0, 255, 0), 2)
                        blur_region(image_blurred, top_left, bottom_right, blur_intensity)

                        output_data["english_text"].append(text)
                        output_data["english_text_coordinates"].append({
                            "top_left": top_left,
                            "bottom_right": bottom_right
                        })

         
                json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_output.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)

               
                detected_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_detected.jpg")
                blurred_image_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_blurred.jpg")
                cv2.imwrite(detected_image_path, image_detected)
                cv2.imwrite(blurred_image_path, image_blurred)

                print(f"Processed {filename} - Results saved in {output_dir}")

# Usage
directory_path = 'C:\\Users\\Rohit\\OneDrive\\Desktop\\YOVA MBZ\\OneDrive_2024-10-29\\Abu Dhabi - Updated'  
process_images_in_directory(directory_path, blur_intensity=101)
