"""Abstract: The goal of this this proejct is to short through the entire kaggle dataset https://www.kaggle.com/c/airbus-ship-detection 
I want t go through this and devlope a labeleed dataset so that I can go aehad 
# I have already used the dynamic feature vectors Ubsupevisied learning algo to group simliar images.
# I need to start parsing through all of these images and gather ships that I can label. into one folde and all other images in a sperate locations
#


"""
import cv2
from pathlib import Path
from typing import Any
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
#import dynamic_feature_vector_builder as dfvb
import yaml
from matplotlib import pyplot as plt
import re
import shutil
class finding_imgs_and_bounding_boxes():

    def __init__(self):
        pass
        

    def canny_ship_detection(self, img_path)-> bool | list:
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img , (5,5) , 0 )

        t_upper = 50
        t_lower = 150

        edge = cv2.Canny(blurred , 50 , 150)

        contours, _ = cv2.findContours(blurred , t_upper , t_lower)

        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
            else : 
                ship_found = False
                bbx = None

        bbx = [x, y, w, h] 
        ship_found=True
        return ship_found ,bbx
    
    #todo
    #get rid of the various _boxed , enure txt and the file have the same same. 

    def manually_draw_BBXs(self, img_path):
        img_path = Path(img_path)
        original = cv2.imread(str(img_path))

        if original is None:
            raise ValueError(f"Could not read image: {img_path}")

        display_img = original.copy()  # this one gets boxes drawn on it for the UI

        output_dir = Path(r"D:\Portfolio\Ships\Final_data_set")
        output_dir.mkdir(parents=True, exist_ok=True)

        img_dir = output_dir / "images"
        label_dir = output_dir / "labels"

        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        boxes = []
        ref_points = []

        h_img, w_img = original.shape[:2]

        def mouse_callback(event, x, y, flags, param):
            nonlocal display_img, ref_points, boxes

            if event == cv2.EVENT_LBUTTONDOWN:
                ref_points = [(x, y)]

            elif event == cv2.EVENT_LBUTTONUP:
                ref_points.append((x, y))

                x1, y1 = ref_points[0]
                x2, y2 = ref_points[1]

                x_min = min(x1, x2)
                y_min = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                if w == 0 or h == 0:
                    return

                boxes.append([x_min, y_min, w, h])

                cv2.rectangle(
                    display_img,
                    (x_min, y_min),
                    (x_min + w, y_min + h),
                    (0, 255, 0),
                    2
                )

                cv2.imshow("image", display_img)

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_callback)

        cv2.imshow("image", display_img)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC -- finish, save image + labels
                break

            elif key in (ord("c"), ord("C")):  # clear all boxes, start fresh
                boxes.clear()
                ref_points.clear()
                display_img = original.copy()
                cv2.imshow("image", display_img)

        cv2.destroyAllWindows()

        # Save the ORIGINAL, unmodified image
        saved_img_path = img_dir / f"{img_path.stem}{img_path.suffix}"
        cv2.imwrite(str(saved_img_path), original)

        # Save YOLO label file
        label_path = label_dir / f"{img_path.stem}.txt"

        with open(label_path, "w") as f:
            for box in boxes:
                x_min, y_min, w, h = box

                x_center = (x_min + w / 2) / w_img
                y_center = (y_min + h / 2) / h_img
                box_width = w / w_img
                box_height = h / h_img

                class_id = 0  # ship

                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}\n"
                )

        return {
            "boxes": boxes,
            "saved_image": str(saved_img_path),
            "label_file": str(label_path),
            "ship_count": len(boxes)
    }


class TrainYolo:
    def __init__(self,training_data:Path = None)  :
        self.training_data = training_data


    @staticmethod
    def train_YOLO_MODEL():
        model = YOLO("yolo11n.pt")
        dataset_path = r'D:\Portfolio\Ships\Final_data_set'
        model.train(
                data=f"{dataset_path}/ships.yaml",
                epochs=200,
                imgsz=768,
                project=dataset_path,
                name="training_results",
                exist_ok=True
            )
        model.save("computervision/Unspervised learning/Porfolio_Projects/USPL/custom_model_final_draft.pt")

    
    def create_yolo_yaml(self):
        yaml_data = {
            "path": str(self.training_data),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {
                0: "ship"
            }
        }

        yaml_path = self.training_data / "ships.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

        print(f"Created: {yaml_path}")
    @staticmethod
    def load_trained_yolo_model():
        return YOLO(r"C:\Python Stuff\Programming_Folder\computervision\Unspervised learning\Porfolio_Projects\USPL\custom_model_final_draft.pt")

    def plot_boxes(self , results) ->list[int] | bool | Path:
        img = results[0].orig_img.copy()
        names = results[0].names
        scores = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
        for score, cls, bbox in zip(scores, classes, boxes):
            class_label = names[int(cls)]
            label = f"{class_label}: {score:.2f}"

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color=(0, 255, 0),
                thickness=2
            )

            lbl_margin = 3
            (lbl_w, lbl_h), _ = cv2.getTextSize(
                label,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                thickness=1
            )

            cv2.rectangle(
                img,
                (x1, y1),
                (x1 + lbl_w + 2 * lbl_margin, y1 - lbl_h - 2 * lbl_margin),
                color=(0, 255, 0),
                thickness=-1
            )

            cv2.putText(
                img,
                label,
                (x1 + lbl_margin, y1 - lbl_margin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=1
            )

        return img
"""     draw_bbx = finding_imgs_and_bounding_boxes()

    original_root = Path(r"D:\Portfolio\Ships\sorted\CLIP\Sequestered_Ships")
    txt_root = Path(r"D:\Portfolio\Ships\sorted\CLIP\Sequestered_Ships\Result_data")
    boxed_root = Path(r"D:\Portfolio\Ships\sorted\CLIP\Sequestered_Ships\Boxed_imgs")

    # Matches where manually_draw_BBXs saves, so the whole final dataset
    # lands in the images/ + labels/ layout YOLO expects
    dest_img = Path(r"D:\Portfolio\Ships\Final_data_set\images")
    dest_text = Path(r"D:\Portfolio\Ships\Final_data_set\labels")

    dest_img.mkdir(parents=True, exist_ok=True)
    dest_text.mkdir(parents=True, exist_ok=True)

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    quit_review = False

    for boxed_file in sorted(boxed_root.iterdir()):
        if boxed_file.suffix.lower() not in IMG_EXTS:
            continue

        # Strip the _boxed suffix to recover the shared stem,
        # e.g. 00a9be085_boxed.png -> 00a9be085
        stem = boxed_file.stem.replace("_boxed", "")

        original_path = original_root / f"{stem}{boxed_file.suffix}"
        txt_path = txt_root / f"{stem}.txt"

        if not original_path.exists() or not txt_path.exists():
            print(f"Skipping {boxed_file.name}: matching original image or txt not found")
            continue

        img = cv2.imread(str(boxed_file))
        if img is None:
            print(f"Skipping {boxed_file.name}: could not read image")
            continue

        cv2.imshow("review", img)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                # Keep: YOLO's boxes were good -- move original + label to final dataset
                shutil.move(str(original_path), str(dest_img / original_path.name))
                shutil.move(str(txt_path), str(dest_text / txt_path.name))
                boxed_file.unlink()
                break

            elif key in (ord("d"), ord("D")):
                # Reject: not a ship / bad crop -- delete all three files
                txt_path.unlink()
                original_path.unlink()
                boxed_file.unlink()
                break

            elif key in (ord("r"), ord("R")):
                # Redraw: YOLO's boxes were wrong -- label it by hand instead.
                # manually_draw_BBXs saves a fresh copy of the image + new label
                # into Final_data_set, so the stale source files can go.
                draw_bbx.manually_draw_BBXs(img_path=original_path)
                txt_path.unlink()
                original_path.unlink()
                boxed_file.unlink()
                break

            elif key == 27:  # ESC -- stop the session; progress is preserved
                quit_review = True
                break

            elif key != 255:  # 255 = no key pressed on this tick
                print("Invalid key -- s = save, d = delete, r = redraw, ESC = quit")

        if quit_review:
            break

    cv2.destroyAllWindows()
 """



if __name__ == "__main__":
    # Test first without deleting anything.
    draw_bbx = finding_imgs_and_bounding_boxes()

    original_root = Path(r"D:\Portfolio\Ships\sorted\CLIP\Sequestered_Ships")
    txt_root = Path(r"D:\Portfolio\Ships\sorted\CLIP\Sequestered_Ships\Result_data")
    boxed_root = Path(r"D:\Portfolio\Ships\sorted\CLIP\Sequestered_Ships\Boxed_imgs")

    # Matches where manually_draw_BBXs saves, so the whole final dataset
    # lands in the images/ + labels/ layout YOLO expects
    dest_img = Path(r"D:\Portfolio\Ships\Final_data_set\images")
    dest_text = Path(r"D:\Portfolio\Ships\Final_data_set\labels")

    dest_img.mkdir(parents=True, exist_ok=True)
    dest_text.mkdir(parents=True, exist_ok=True)

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    quit_review = False

    for boxed_file in sorted(boxed_root.iterdir()):
        if boxed_file.suffix.lower() not in IMG_EXTS:
            continue

        # Strip the _boxed suffix to recover the shared stem,
        # e.g. 00a9be085_boxed.png -> 00a9be085
        stem = boxed_file.stem.replace("_boxed", "")

        original_path = original_root / f"{stem}{boxed_file.suffix}"
        txt_path = txt_root / f"{stem}.txt"

        if not original_path.exists() or not txt_path.exists():
            print(f"Skipping {boxed_file.name}: matching original image or txt not found")
            continue

        img = cv2.imread(str(boxed_file))
        if img is None:
            print(f"Skipping {boxed_file.name}: could not read image")
            continue

        cv2.imshow("review", img)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                # Keep: YOLO's boxes were good -- move original + label to final dataset
                shutil.move(str(original_path), str(dest_img / original_path.name))
                shutil.move(str(txt_path), str(dest_text / txt_path.name))
                boxed_file.unlink()
                break

            elif key in (ord("d"), ord("D")):
                # Reject: not a ship / bad crop -- delete all three files
                txt_path.unlink()
                original_path.unlink()
                boxed_file.unlink()
                break

            elif key in (ord("r"), ord("R")):
                # Redraw: YOLO's boxes were wrong -- label it by hand instead.
                # manually_draw_BBXs saves a fresh copy of the image + new label
                # into Final_data_set, so the stale source files can go.
                draw_bbx.manually_draw_BBXs(img_path=original_path)
                txt_path.unlink()
                original_path.unlink()
                boxed_file.unlink()
                break

            elif key == 27:  # ESC -- stop the session; progress is preserved
                quit_review = True
                break

            elif key != 255:  # 255 = no key pressed on this tick
                print("Invalid key -- s = save, d = delete, r = redraw, ESC = quit")

        if quit_review:
            break

    cv2.destroyAllWindows()

    # After checking the printed paths, use this:
    # holder(dry_run=False)