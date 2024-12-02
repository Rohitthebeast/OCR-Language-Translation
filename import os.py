
import os
import cv2
import matplotlib.pyplot as plt
import easyocr
from bidi.algorithm import get_display
import json
import numpy as np

def merge_english_boxes(boxes, texts, probs, max_distance=20):
    merged_boxes = []
    merged_texts = []
    merged_probs = []

    for i, (box, text, prob) in enumerate(zip(boxes, texts, probs)):
        if i == 0 or not text.isascii():
            merged_boxes.append(box)
            merged_texts.append(text)
            merged_probs.append(prob)
        else:
            prev_box = merged_boxes[-1]
            if merged_texts[-1].isascii():
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
            else:
                merged_boxes.append(box)
                merged_texts.append(text)
                merged_probs.append(prob)

    return merged_boxes, merged_texts, merged_probs

def blur_region(image, tl, br, blur_intensity=51):
    roi = image[tl[1]:br[1], tl[0]:br[0]]

    blur_intensity = blur_intensity if blur_intensity % 2 == 1 else blur_intensity + 1
    blurred_roi = cv2.GaussianBlur(roi, (blur_intensity, blur_intensity), 0)
    image[tl[1]:br[1], tl[0]:br[0]] = blurred_roi

def process_image(image_path, confidence_threshold=0.1, blur_intensity=101):
    reader = easyocr.Reader(['en', 'ar'], gpu=False)

    if os.path.isfile(image_path):
        image = cv2.imread(image_path)

        if image is not None:
            results = reader.readtext(image)

            boxes, texts, probs = zip(*results)
            merged_boxes, merged_texts, merged_probs = merge_english_boxes(boxes, texts, probs)
            merged_results = list(zip(merged_boxes, merged_texts, merged_probs))

            output_data = {
                "image_id": os.path.basename(image_path),
                "english_text": [],
                "english_text_coordinates": []
            }


            image_detected = image.copy()
            image_blurred = image.copy()

            for (bbox, text, prob) in merged_results:
                if text.isascii() and prob > confidence_threshold:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    top_left = tuple(map(int, top_left))
                    bottom_right = tuple(map(int, bottom_right))


                    cv2.rectangle(image_detected, top_left, bottom_right, (0, 255, 0), 2)
                    plt.text(top_left[0], top_left[1] - 10, text, fontsize=12, color='red')


                    blur_region(image_blurred, top_left, bottom_right, blur_intensity)

                    output_data["english_text"].append(text)
                    output_data["english_text_coordinates"].append({
                        "top_left": top_left,
                        "bottom_right": bottom_right
                    })

                    print(f"Detected English text: {text} with confidence: {prob}")


            cv2.imwrite('detected_english.jpg', image_detected)
            cv2.imwrite('blurred_english.jpg', image_blurred)

            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.imshow(cv2.cvtColor(image_detected, cv2.COLOR_BGR2RGB))
            plt.title("Detected English Text")
            plt.axis('off')

            plt.subplot(122)
            plt.imshow(cv2.cvtColor(image_blurred, cv2.COLOR_BGR2RGB))
            plt.title("Blurred English Text")
            plt.axis('off')

            plt.show()


            output_json = json.dumps(output_data, ensure_ascii=False, indent=4)
            with open('output.json', 'w', encoding='utf-8') as f:
                f.write(output_json)
            print(f"\nJSON data saved as output.json")

        else:
            print("Failed to read image")
    else:
        print("Image file does not exist")


image_path = 'Image File'
process_image(image_path, blur_intensity=101)
