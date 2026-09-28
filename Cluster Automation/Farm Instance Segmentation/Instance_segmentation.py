import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2
import math
import random
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
import utils
import random

device  ='cuda'


old_img_name = None
#this is a work in progress to create a class that will load the npy files and allow me to mask them one at a time.
# the masks , boxes and labels need to all be inside of one class . 
class NpyLoadFile:

    def __init__(
        self,
        file_path: Path | None = None,
        root_dir: Path | None = None,
        annotation_root: Path | None = None,
        mask_root: Path | None = None,
        image_root: Path | None = None
    ):

        self.file_path = (
            Path(file_path)
            if file_path is not None
            else None
        )

        self.root_dir = (
            Path(root_dir)
            if root_dir is not None
            else None
        )

        self.annotation_root = (
            Path(annotation_root)
            if annotation_root is not None
            else Path(r"D:\Portfolio\Custom_maks\Annotations")
        )

        self.mask_root = (
            Path(mask_root)
            if mask_root is not None
            else Path(r"D:\Portfolio\Custom_maks\Masks")
        )

        self.image_root = (
            Path(image_root)
            if image_root is not None
            else Path(
                r"D:\Portfolio\PASTIS-R\DATA_S2 - think I want these"
            )
        )
    def load_one_file_all_images(self):
        img_array = np.load(self.file_path)

        number_of_images = img_array.shape[0]

        columns = 7
        rows = math.ceil(number_of_images / columns)

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(18, rows * 2.8),
            squeeze=False
        )

        axes = axes.flatten()

        for index in range(number_of_images):
            rgb = img_array[index, [2, 1, 0], :, :].astype(np.float32)

            channel_min = rgb.min(axis=(1, 2), keepdims=True)
            channel_max = rgb.max(axis=(1, 2), keepdims=True)

            rgb = (
                (rgb - channel_min) /
                (channel_max - channel_min + 1e-8)
            )

            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = np.clip(rgb, 0, 1)

            axes[index].imshow(rgb)
            axes[index].set_title(f"Observation {index}")
            axes[index].axis("off")

        # Hide unused grid positions.
        for index in range(number_of_images, len(axes)):
            axes[index].axis("off")

        fig.suptitle(
            f"{self.file_path.name} — {number_of_images} observations",
            fontsize=16
        )

        plt.tight_layout()
        plt.show()

    def load_entire_dir(self):
        """This loads all of the NPY obersvations and displays them in an pyplot thingy."""
        for file in self.root_dir.iterdir():

            # Skip folders and non-NPY files.
            if not file.is_file() or file.suffix.lower() != ".npy":
                continue

            img_array = np.load(file)
            number_of_images = img_array.shape[0]

            columns = 7
            rows = math.ceil(number_of_images / columns)

            fig, axes = plt.subplots(
                rows,
                columns,
                figsize=(18, rows * 2.8),
                squeeze=False
            )

            axes = axes.flatten()

            for index in range(number_of_images):
                rgb = img_array[
                    index,
                    [2, 1, 0],
                    :,
                    :
                ].astype(np.float32)

                channel_min = rgb.min(
                    axis=(1, 2),
                    keepdims=True
                )

                channel_max = rgb.max(
                    axis=(1, 2),
                    keepdims=True
                )

                rgb = (
                    (rgb - channel_min)
                    / (channel_max - channel_min + 1e-8)
                )

                rgb = np.transpose(rgb, (1, 2, 0))
                rgb = np.clip(rgb, 0, 1)

                # Place the image onto its subplot.
                axes[index].imshow(rgb)
                axes[index].set_title(f"Observation {index}")
                axes[index].axis("off")

            # Hide unused subplot positions.
            for index in range(number_of_images, len(axes)):
                axes[index].axis("off")

            # Use the current NPY filename, not self.file_path.
            fig.suptitle(
                f"{file.name} — {number_of_images} observations",
                fontsize=16
            )

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.show()
            plt.close(fig)
    #this loads aloof the images one at a time so I can mask them more easily
    def annotate_first_apply_to_all(self):

        annotation_path = Path(
            r"D:\Portfolio\Custom_maks\Annotations"
        )

        mask_path = Path(
            r"D:\Portfolio\Custom_maks\Masks"
        )

        annotation_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        annotation_size = 850

        # Load ONLY the file selected in main
        for file_path in [self.file_path]:

            print(f"\nLoading: {file_path.name}")

            img_array = np.load(file_path)

            number_of_images = img_array.shape[0]
            height = img_array.shape[2]
            width = img_array.shape[3]

            # rest of your existing code...

            selected_polygons = None
            selected_observation = None

            # -------------------------------------
            # Search for a usable observation
            # -------------------------------------

            for observation_index in range(1):

                rgb = img_array[
                    observation_index,
                    [2, 1, 0],
                    :,
                    :
                ].astype(np.float32)

                channel_min = rgb.min(
                    axis=(1, 2),
                    keepdims=True
                )

                channel_max = rgb.max(
                    axis=(1, 2),
                    keepdims=True
                )

                rgb = (
                    (rgb - channel_min)
                    /
                    (channel_max - channel_min + 1e-8)
                )

                rgb = np.transpose(
                    rgb,
                    (1, 2, 0)
                )

                rgb = np.clip(
                    rgb,
                    0,
                    1
                )

                display_rgb = cv2.resize(
                    rgb,
                    (annotation_size, annotation_size),
                    interpolation=cv2.INTER_LANCZOS4
                )

                print(
                    f"{file_path.name} | "
                    f"Observation "
                    f"{observation_index}/{number_of_images - 1}"
                )
                result = create_mask.draw_polygons(
                    rgb=display_rgb,
                    file_name=file_path.name,
                    observation_index=observation_index
                )

                if result is None:
                    print(f"Skipping observation {observation_index}")
                    continue

                polygons, annotation_save_path = result
                # ESC → skip cloudy/bad observation
                if polygons is None:

                    print(
                        f"Skipping observation "
                        f"{observation_index}"
                    )

                    continue

                # Q → accept this observation
                selected_polygons = polygons
                selected_observation = observation_index

                print(
                    f"Using observation "
                    f"{selected_observation}"
                )

                break

            # -------------------------------------
            # No usable image
            # -------------------------------------

            if selected_polygons is None:

                print(
                    f"No usable observation found "
                    f"for {file_path.name}"
                )

                continue

            # -------------------------------------
            # Convert selected polygons ONCE
            # -------------------------------------

            boxes, labels, masks = (
                create_mask.create_training_targets(
                    polygons=selected_polygons,
                    display_size=(
                        annotation_size,
                        annotation_size
                    ),
                    native_size=(
                        width,
                        height
                    )
                )
            )

            print(
                f"Created {len(labels)} instances "
                f"from observation "
                f"{selected_observation}"
            )

            # -------------------------------------
            # Apply them to every observation
            # -------------------------------------

            mask_save_path = (
                mask_path
                / annotation_save_path.name
            )

            mask_save_path.mkdir(
                parents=True,
                exist_ok=True
            )

            self.save_annotations_for_all_observations(
                file_path=file_path,
                number_of_images=number_of_images,
                boxes=boxes,
                labels=labels,
                masks=masks,
                annotation_path=annotation_save_path,
                mask_path=mask_save_path
            )
    @staticmethod
    def save_annotations_for_all_observations(
        file_path: Path,
        number_of_images: int,
        boxes: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray,
        annotation_path: Path,
        mask_path: Path
    ):

        for observation_index in range(number_of_images):

            annotation_name = (
                f"{file_path.stem}_"
                f"obs_{observation_index:03d}"
            )

            np.save(
                annotation_path
                / f"{annotation_name}_boxes.npy",
                boxes
            )

            np.save(
                annotation_path
                / f"{annotation_name}_labels.npy",
                labels
            )

            np.save(
                mask_path
                / f"{annotation_name}_masks.npy",
                masks
            )

        print(
            f"Applied annotations to "
            f"{number_of_images} observations."
        )

    def validate_annotations(self):

        for split in ["Train", "Test", "Val"]:

            annotation_dir = self.annotation_root / split
            mask_dir = self.mask_root / split

            print(f"\nValidating {split}")

            # Only loop over boxes so each observation
            # is reviewed ONE time.
            for boxes_path in sorted(
                annotation_dir.glob("*_boxes.npy")
            ):

                base_name = boxes_path.stem.replace(
                    "_boxes",
                    ""
                )

                labels_path = (
                    annotation_dir
                    / f"{base_name}_labels.npy"
                )

                masks_path = (
                    mask_dir
                    / f"{base_name}_masks.npy"
                )

                # Example:
                # S2_10000_obs_003
                match = re.match(
                    r"(S2_\d+)_obs_(\d+)",
                    base_name
                )

                if match is None:
                    print(f"Bad filename: {base_name}")
                    continue

                image_id = match.group(1)
                observation_index = int(
                    match.group(2)
                )

                image_path = (
                    self.image_root
                    / f"{image_id}.npy"
                )

                if not image_path.exists():
                    print(
                        f"Missing original image: "
                        f"{image_path}"
                    )
                    continue

                if not masks_path.exists():
                    print(
                        f"Missing mask: "
                        f"{masks_path}"
                    )
                    continue

                action = create_mask.validate_data(
                    image_path=image_path,
                    observation_index=observation_index,
                    boxes_path=boxes_path,
                    labels_path=labels_path,
                    masks_path=masks_path
                )

                # ESC exits the whole validation process.
                if action == "exit":
                    print("Validation stopped.")
                    return
                    
class create_mask():
    def __init__(self, mask_path:Path , annotaions_maks:list):
        self.mask_path = mask_path
        self.annotaions_maks = [] or annotaions_maks
    @staticmethod
    def validate_data(
            image_path: Path,
            observation_index: int,
            boxes_path: Path,
            labels_path: Path,
            masks_path: Path
        ):

            # -----------------------------------
            # Load original satellite observation
            # -----------------------------------

            img_array = np.load(image_path)

            rgb = img_array[
                observation_index,
                [2, 1, 0],
                :,
                :
            ].astype(np.float32)

            height = rgb.shape[1]
            width = rgb.shape[2]

            channel_min = rgb.min(
                axis=(1, 2),
                keepdims=True
            )

            channel_max = rgb.max(
                axis=(1, 2),
                keepdims=True
            )

            rgb = (
                (rgb - channel_min)
                /
                (channel_max - channel_min + 1e-8)
            )

            rgb = np.transpose(
                rgb,
                (1, 2, 0)
            )

            rgb = np.clip(
                rgb,
                0,
                1
            )

            # -----------------------------------
            # Load existing targets
            # -----------------------------------

            boxes = np.load(boxes_path)

            if labels_path.exists():
                labels = np.load(labels_path)
            else:
                labels = np.empty(
                    (0,),
                    dtype=np.int64
                )

            masks = np.load(masks_path)

            # -----------------------------------
            # Build display image
            # -----------------------------------

            display = (
                rgb * 255
            ).astype(np.uint8)

            display = cv2.cvtColor(
                display,
                cv2.COLOR_RGB2BGR
            )

            display_size = 850

            display = cv2.resize(
                display,
                (display_size, display_size),
                interpolation=cv2.INTER_LANCZOS4
            )

            scale_x = display_size / width
            scale_y = display_size / height

            # -----------------------------------
            # Draw current masks
            # -----------------------------------

            for mask in masks:

                resized_mask = cv2.resize(
                    mask.astype(np.uint8),
                    (display_size, display_size),
                    interpolation=cv2.INTER_NEAREST
                )

                contours, _ = cv2.findContours(
                    resized_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                cv2.drawContours(
                    display,
                    contours,
                    -1,
                    (0, 255, 0),
                    2
                )

            # -----------------------------------
            # Draw current boxes
            # -----------------------------------

            for box in boxes:

                x1, y1, x2, y2 = box

                cv2.rectangle(
                    display,
                    (
                        int(x1 * scale_x),
                        int(y1 * scale_y)
                    ),
                    (
                        int(x2 * scale_x),
                        int(y2 * scale_y)
                    ),
                    (255, 0, 0),
                    1
                )

            window_name = "Annotation Validator"

            cv2.namedWindow(
                window_name,
                cv2.WINDOW_AUTOSIZE
            )

            cv2.imshow(
                window_name,
                display
            )

            print(
                f"\n{boxes_path.name}"
            )

            print(
                "SPACE = keep | "
                "C = clear cloudy image | "
                "Z = redraw | "
                "ESC = quit"
            )

            # Wait until user chooses.
            while True:

                key = cv2.waitKey(0) & 0xFF

                # -----------------------------------
                # SPACE -> annotation is good
                # -----------------------------------

                if key == ord(" "):

                    cv2.destroyWindow(
                        window_name
                    )

                    return "keep"

                # -----------------------------------
                # C -> CLEAR ALL TARGETS
                # -----------------------------------

                elif key == ord("c"):

                    empty_boxes = np.empty(
                        (0, 4),
                        dtype=np.float32
                    )

                    empty_labels = np.empty(
                        (0,),
                        dtype=np.int64
                    )

                    empty_masks = np.empty(
                        (0, height, width),
                        dtype=np.uint8
                    )

                    np.save(
                        boxes_path,
                        empty_boxes
                    )

                    np.save(
                        labels_path,
                        empty_labels
                    )

                    np.save(
                        masks_path,
                        empty_masks
                    )

                    print(
                        f"Cleared annotation: "
                        f"{boxes_path.stem}"
                    )

                    cv2.destroyWindow(
                        window_name
                    )

                    return "cleared"

                # -----------------------------------
                # Z -> REDRAW
                # -----------------------------------

                elif key == ord("z"):

                    cv2.destroyWindow(
                        window_name
                    )

                    polygons = (
                        create_mask.redraw_polygons(
                            rgb=rgb
                        )
                    )

                    if polygons is None:
                        print(
                            "Redraw cancelled."
                        )
                        return "keep"

                    boxes, labels, masks = (
                        create_mask.create_training_targets(
                            polygons=polygons,
                            display_size=(
                                850,
                                850
                            ),
                            native_size=(
                                width,
                                height
                            )
                        )
                    )

                    # OVERWRITE old targets
                    np.save(
                        boxes_path,
                        boxes
                    )

                    np.save(
                        labels_path,
                        labels
                    )

                    np.save(
                        masks_path,
                        masks
                    )

                    print(
                        f"Redrew {len(labels)} "
                        f"instances."
                    )

                    return "redrawn"

                # -----------------------------------
                # ESC -> quit everything
                # -----------------------------------

                elif key == 27:

                    cv2.destroyAllWindows()

                    return "exit"
    @staticmethod
    def draw_polygons(
        rgb: np.ndarray,
        file_name: str,
        observation_index: int
    ):
        polygons: list[list[tuple[int, int]]] = []
        current_polygon: list[tuple[int, int]] = []
        current_mouse_position = None

        # Use the same window for every image.
        window_name = "Parcel Annotation"

        display_image = np.clip(
            rgb * 255,
            0,
            255
        ).astype(np.uint8)

        display_image = cv2.cvtColor(
            display_image,
            cv2.COLOR_RGB2BGR
        )
        def mouse_callback(event, x, y, flags, param):

            nonlocal polygons
            nonlocal current_polygon
            nonlocal current_mouse_position

            if event == cv2.EVENT_MOUSEMOVE:
                current_mouse_position = (x, y)

            elif event == cv2.EVENT_LBUTTONDOWN:
                current_polygon.append((x, y))
                current_mouse_position = (x, y)

            elif event == cv2.EVENT_RBUTTONDOWN:

                if len(current_polygon) >= 3:
                    polygons.append(
                        current_polygon.copy()
                    )

                current_polygon.clear()
                current_mouse_position = None
                

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.setMouseCallback(
            window_name,
            mouse_callback
        )
        print(  "Left click: point | "
                    "Right click: finish parcel | "
                    "C: clear current | "
                    "U: undo | "
                    "Q: save"
                )

        while True:
            canvas = display_image.copy()
            canvas = cv2.resize(canvas, (850, 850), interpolation=cv2.INTER_LANCZOS4)

            for polygon in polygons:
                polygon_array = np.asarray(
                    polygon,
                    dtype=np.int32
                )

                cv2.polylines(
                    canvas,
                    [polygon_array],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2
                )

            # Draw the polygon currently being created.
            if current_polygon:
                current_array = np.asarray(
                    current_polygon,
                    dtype=np.int32
                )

                cv2.polylines(
                    canvas,
                    [current_array],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=2
                )

                for point in current_polygon:
                    cv2.circle(
                        canvas,
                        point,
                        radius=4,
                        color=(0, 0, 255),
                        thickness=-1
                    )

                if current_mouse_position is not None:
                    cv2.line(
                        canvas,
                        current_polygon[-1],
                        current_mouse_position,
                        color=(255, 255, 0),
                        thickness=1
                    )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("c"):
                current_polygon.clear()

            elif key == ord("u"):
                if current_polygon:
                    current_polygon.pop()
                elif polygons:
                    polygons.pop()

            elif key == ord("q"):
                # Automatically finish the current polygon.
                if len(current_polygon) >= 3:
                    polygons.append(current_polygon.copy())
                    current_polygon.clear()
                    #this is how I randomly distribute the images into train , test , val
                num1 = random.randint(1, 100)

                if num1 <= 50:
                    print("Training Data")
                    annotation_save_path = Path(
                        r'D:\Portfolio\Custom_maks\Annotations\Train'
                    )

                elif num1 <= 75:
                    print("Test Data")
                    annotation_save_path = Path(
                        r'D:\Portfolio\Custom_maks\Annotations\Test'
                    )

                else:
                    print("Val Data")
                    annotation_save_path = Path(
                        r'D:\Portfolio\Custom_maks\Annotations\Val'
                    )

                break

            elif key == 27:
                cv2.destroyWindow(window_name)
                return None

        cv2.destroyWindow(window_name)

        return polygons , annotation_save_path
    
    @staticmethod
    def create_training_targets(
        polygons: list[list[tuple[int, int]]],
        display_size: tuple[int, int] = (300, 300),
        native_size: tuple[int, int] = (128, 128)
    ):
        display_width, display_height = display_size
        native_width, native_height = native_size

        instance_masks = []
        boxes = []

        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.float32)

            if points.shape[0] < 3:
                continue

            # Convert 600x600 display coordinates back to 128x128.
            points[:, 0] *= native_width / display_width
            points[:, 1] *= native_height / display_height

            points = np.rint(points).astype(np.int32)

            # Keep coordinates inside the native image.
            points[:, 0] = np.clip(points[:, 0], 0, native_width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, native_height - 1)

            # One 2D binary mask per parcel.
            mask = np.zeros(
                (native_height, native_width),
                dtype=np.uint8
            )

            cv2.fillPoly(
                mask,
                [points],
                1
            )

            ys, xs = np.where(mask == 1)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x_min = xs.min()
            y_min = ys.min()
            x_max = xs.max() + 1
            y_max = ys.max() + 1

            boxes.append([
                x_min,
                y_min,
                x_max,
                y_max
            ])

            instance_masks.append(mask)

        number_of_instances = len(instance_masks)

        if number_of_instances == 0:
            boxes_array = np.empty((0, 4), dtype=np.float32)
            labels_array = np.empty((0,), dtype=np.int64)
            masks_array = np.empty(
                (0, native_height, native_width),
                dtype=np.uint8
            )
        else:
            boxes_array = np.asarray(boxes, dtype=np.float32)

            labels_array = np.ones(
                number_of_instances,
                dtype=np.int64
            )

            masks_array = np.stack(
                instance_masks,
                axis=0
            ).astype(np.uint8)

        return boxes_array, labels_array, masks_array

    def annoate_other_imgs(self):
        if self.annotaions_maks is not None:

            for AM in self.annotaions_maks:
                pts = np.array(AM,np.int32)
                pts = pts.reshape((-1 , 1 , 2))
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2
                isClosed = True
                # Using cv2.polylines() method
                # Draw a Blue polygon with 
                # thickness of 1 px
                image = cv2.polylines(image, [pts], 
                                    isClosed, color, thickness)
    
    @staticmethod
    def redraw_polygons(
        rgb: np.ndarray
    ):

        polygons = []
        current_polygon = []
        current_mouse_position = None

        window_name = "Redraw Annotation"

        display_image = np.clip(
            rgb * 255,
            0,
            255
        ).astype(np.uint8)

        display_image = cv2.cvtColor(
            display_image,
            cv2.COLOR_RGB2BGR
        )

        display_image = cv2.resize(
            display_image,
            (850, 850),
            interpolation=cv2.INTER_LANCZOS4
        )

        def mouse_callback(
            event,
            x,
            y,
            flags,
            param
        ):

            nonlocal polygons
            nonlocal current_polygon
            nonlocal current_mouse_position

            if event == cv2.EVENT_MOUSEMOVE:

                current_mouse_position = (
                    x,
                    y
                )

            elif event == cv2.EVENT_LBUTTONDOWN:

                current_polygon.append(
                    (x, y)
                )

            elif event == cv2.EVENT_RBUTTONDOWN:

                if len(current_polygon) >= 3:

                    polygons.append(
                        current_polygon.copy()
                    )

                current_polygon.clear()

        cv2.namedWindow(
            window_name
        )

        cv2.setMouseCallback(
            window_name,
            mouse_callback
        )

        print(
            "Left click = point | "
            "Right click = finish parcel | "
            "U = undo | "
            "C = clear current | "
            "Q = save | "
            "ESC = cancel"
        )

        while True:

            canvas = display_image.copy()

            for polygon in polygons:

                pts = np.asarray(
                    polygon,
                    dtype=np.int32
                )

                cv2.polylines(
                    canvas,
                    [pts],
                    True,
                    (0, 255, 0),
                    2
                )

            if current_polygon:

                pts = np.asarray(
                    current_polygon,
                    dtype=np.int32
                )

                cv2.polylines(
                    canvas,
                    [pts],
                    False,
                    (0, 255, 255),
                    2
                )

                for point in current_polygon:

                    cv2.circle(
                        canvas,
                        point,
                        4,
                        (0, 0, 255),
                        -1
                    )

                if (
                    current_mouse_position
                    is not None
                ):

                    cv2.line(
                        canvas,
                        current_polygon[-1],
                        current_mouse_position,
                        (255, 255, 0),
                        1
                    )

            cv2.imshow(
                window_name,
                canvas
            )

            key = cv2.waitKey(20) & 0xFF

            if key == ord("c"):

                current_polygon.clear()

            elif key == ord("u"):

                if current_polygon:

                    current_polygon.pop()

                elif polygons:

                    polygons.pop()

            elif key == ord("q"):

                if len(current_polygon) >= 3:

                    polygons.append(
                        current_polygon.copy()
                    )

                if not polygons:

                    print(
                        "No polygons drawn."
                    )

                    continue

                cv2.destroyWindow(
                    window_name
                )

                return polygons

            elif key == 27:

                cv2.destroyWindow(
                    window_name
                )

                return None
        
class train_mask_rcnn():
    def __init__(self, model_path:Path , train_data_path:Path , test_data_path:Path , val_data_path:Path, original_imgs:Path, 
                 maks_train:Path , maks_test:Path , maks_val:Path, annotation_Keyword:str,labels_keyword:str) :
        self.model_path = model_path
        self.train_data_path = train_data_path or r'D:\Portfolio\Custom_maks\Annotations\Train'
        self.test_data_path = test_data_path or r'D:\Portfolio\Custom_maks\Annotations\Test'
        self.val_data_path = val_data_path or r'D:\Portfolio\Custom_maks\Annotations\Val'
        self.original_imgs = original_imgs or r'D:\Portfolio\PASTIS-R\path_2'
        self.annotation_Keyword = annotation_Keyword  or "boxes"
        self.labels_keyword = labels_keyword or "labels"

        self.maks_train = maks_train or r'D:\Portfolio\Custom_maks\Masks\Train'
        self.maks_test = maks_test or r'D:\Portfolio\Custom_maks\Masks\Test'
        self.maks_val = maks_val or r'D:\Portfolio\Custom_maks\Masks\Val'

    # 3 lists of dicts with train , val and test data
    def loading_dataset(self):

        dataset = {
            "Train": [],
            "Test": [],
            "Val": []
        }

        paths_to_loop_through = {
            "Train": Path(self.train_data_path),
            "Test": Path(self.test_data_path),
            "Val": Path(self.val_data_path)
        }

        mask_paths = {
            "Train": Path(self.maks_train),
            "Test": Path(self.maks_test),
            "Val": Path(self.maks_val)
        }

        pattern = r"(S2_\d+)_obs_(\d+)_boxes\.npy"

        for split_name, split_path in paths_to_loop_through.items():

            all_boxes = [
                file for file in split_path.iterdir()
                if self.annotation_Keyword in file.name
            ]

            for box_path in all_boxes:

                match = re.match(pattern, box_path.name)

                if not match:
                    continue

                image_id = match.group(1)
                observation_index = int(match.group(2))

                original_path = (
                    Path(self.original_imgs) /
                    f"{image_id}.npy"
                )

                labels_path = (
                    split_path /
                    f"{image_id}_obs_{observation_index:03d}_labels.npy"
                )

                mask_path = (
                    mask_paths[split_name] /
                    f"{image_id}_obs_{observation_index:03d}_masks.npy"
                )

                boxes = np.load(box_path)
                labels = np.load(labels_path)
                masks = np.load(mask_path)

                original_array = np.load(original_path)

                original_image = original_array[observation_index]

                sample = {
                    "image_id": image_id,
                    "observation_index": observation_index,
                    "image": original_image,
                    "boxes": boxes,
                    "labels": labels,
                    "masks": masks
                }

                dataset[split_name].append(sample)

        return dataset


    @staticmethod
    def define_transformation(dataset: dict[str, list[dict]]):

        for split_name, samples in dataset.items():

            for d in samples:

                image = d["image"]
                boxes = d["boxes"]
                labels = d["labels"]
                masks = d["masks"]

                # Select RGB bands: [C, H, W]
                image = image[[2, 1, 0], :, :]

                # Convert NumPy arrays to tensors
                image_tensor = torch.from_numpy(image).float()
                boxes_tensor = torch.from_numpy(boxes).float()
                labels_tensor = torch.from_numpy(labels).long()
                masks_tensor = torch.from_numpy(masks).to(torch.uint8)

                # Normalize each image channel independently to 0-1
                channel_min = image_tensor.amin(
                    dim=(1, 2),
                    keepdim=True
                )

                channel_max = image_tensor.amax(
                    dim=(1, 2),
                    keepdim=True
                )

                image_tensor = (
                    image_tensor - channel_min
                ) / (
                    channel_max - channel_min + 1e-8
                )

                # Store tensors back in sample
                d["img_tensor"] = image_tensor
                d["boxes_tensor"] = boxes_tensor
                d["labels_tensor"] = labels_tensor
                d["masks_tensor"] = masks_tensor

        return dataset

class Training_the_model(train_mask_rcnn):

    def __init__(
        self,
        dataset,
        model_path=None,
        train_data_path=None,
        test_data_path=None,
        val_data_path=None,
        original_imgs=None,
        maks_train=None,
        maks_test=None,
        maks_val=None,
        annotation_Keyword="boxes",
        labels_keyword="labels"
    ):

        super().__init__(
            model_path,
            train_data_path,
            test_data_path,
            val_data_path,
            original_imgs,
            maks_train,
            maks_test,
            maks_val,
            annotation_Keyword,
            labels_keyword
        )

        self.dataset = dataset
    def train_model(self, num_epochs=10, batch_size=2):
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        train_data = self.dataset["Train"]

        def collate_fn(batch):
            return batch

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Load pretrained Mask R-CNN
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights="DEFAULT"
        )

        num_classes = 2  # background + parcel

        # Replace bounding-box predictor
        in_features = (
            model.roi_heads.box_predictor.cls_score.in_features
        )

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes
        )

        # Replace mask predictor
        in_features_mask = (
            model.roi_heads.mask_predictor.conv5_mask.in_channels
        )

        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes
        )

        model.to(device)

        params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        for epoch in range(num_epochs):

            model.train()

            total_loss = 0

            for batch in train_loader:

                images = [
                    sample["img_tensor"].to(device)
                    for sample in batch
                ]

                targets = []

                for sample in batch:

                    target = {
                        "boxes":
                            sample["boxes_tensor"].to(device),

                        "labels":
                            sample["labels_tensor"].to(device),

                        "masks":
                            sample["masks_tensor"].to(device)
                    }

                    targets.append(target)

                optimizer.zero_grad()

                loss_dict = model(images, targets)

                losses = sum(
                    loss for loss in loss_dict.values()
                )

                losses.backward()

                optimizer.step()

                total_loss += losses.item()

            lr_scheduler.step()

            average_loss = total_loss / len(train_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"Loss: {average_loss:.4f}"
            )

            torch.save(
                model.state_dict(),
                f"mask_rcnn_epoch_{epoch + 1}.pth"
            )

        return model
    def noisy_student(self, model):

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model.to(device)
        model.eval()

        return model
    def more_training_Data(
        self,
        model,
        confidence_threshold=0.50,
        annotation_path=r"D:\Portfolio\Custom_maks\Annotations",
        mask_path=r"D:\Portfolio\Custom_maks\Masks"
    ):

        # ---------------------------------------------------------
        # Validate threshold
        # ---------------------------------------------------------

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

        # ---------------------------------------------------------
        # Output directories
        # ---------------------------------------------------------

        annotation_path = Path(
            annotation_path
        )

        mask_path = Path(
            mask_path
        )

        annotation_path.mkdir(
            parents=True,
            exist_ok=True
        )

        mask_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model.to(device)
        model.eval()

        # ---------------------------------------------------------
        # Let OUR confidence threshold control filtering.
        #
        # Torchvision will still perform NMS and limit the number
        # of detections, but we apply the actual score threshold
        # ourselves below.
        # ---------------------------------------------------------

        if hasattr(
            model.roi_heads,
            "score_thresh"
        ):
            model.roi_heads.score_thresh = 0.0

        review_window = (
            "Noisy Student Review"
        )

        print(
            f"\nConfidence threshold: "
            f"{confidence_threshold:.2f}"
        )

        # ---------------------------------------------------------
        # Loop through original NPY files
        # ---------------------------------------------------------

        for file in Path(
            self.original_imgs
        ).iterdir():

            if not file.is_file():
                continue

            if file.suffix.lower() != ".npy":
                continue

            print(
                "\n" + "=" * 60
            )

            print(
                f"Processing: {file.name}"
            )

            print(
                "=" * 60
            )

            # -----------------------------------------------------
            # Load complete temporal sequence
            # -----------------------------------------------------

            img_array = np.load(
                file
            )

            number_of_images = (
                img_array.shape[0]
            )

            # Example:
            # S2_10033.npy -> S2_10033

            image_id = file.stem


            for observation_index in range(
                1
            ):

                image = (
                    img_array[
                        observation_index
                    ]
                )

                # -------------------------------------------------
                # Select RGB bands
                #
                # [C, H, W]
                # -------------------------------------------------

                rgb = image[
                    [2, 1, 0],
                    :,
                    :
                ].astype(
                    np.float32
                )

                height = rgb.shape[1]
                width = rgb.shape[2]

                # -------------------------------------------------
                # SAME NORMALIZATION USED DURING TRAINING
                # -------------------------------------------------

                channel_min = rgb.min(
                    axis=(1, 2),
                    keepdims=True
                )

                channel_max = rgb.max(
                    axis=(1, 2),
                    keepdims=True
                )

                rgb_normalized = (
                    rgb - channel_min
                ) / (
                    channel_max
                    - channel_min
                    + 1e-8
                )

                rgb_normalized = np.clip(
                    rgb_normalized,
                    0,
                    1
                )

                # -------------------------------------------------
                # NumPy -> Tensor
                # -------------------------------------------------

                image_tensor = (
                    torch.from_numpy(
                        rgb_normalized
                    )
                    .float()
                    .to(device)
                )

                # -------------------------------------------------
                # MASK R-CNN INFERENCE
                # -------------------------------------------------

                with torch.inference_mode():

                    prediction = model(
                        [image_tensor]
                    )[0]

                # -------------------------------------------------
                # CONFIDENCE FILTER
                # -------------------------------------------------

                keep = (
                    prediction["scores"]
                    >= confidence_threshold
                )

                boxes = (
                    prediction["boxes"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                )

                labels = (
                    prediction["labels"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                )

                scores = (
                    prediction["scores"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                )

                mask_probabilities = (
                    prediction["masks"][keep]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # -------------------------------------------------
                # Convert probability masks into binary masks
                #
                # [N, 1, H, W]
                # ->
                # [N, H, W]
                # -------------------------------------------------

                if len(
                    mask_probabilities
                ) > 0:

                    predicted_masks = (
                        mask_probabilities[
                            :, 0
                        ]
                        >= 0.5
                    ).astype(
                        np.uint8
                    )

                else:

                    predicted_masks = (
                        np.empty(
                            (
                                0,
                                height,
                                width
                            ),
                            dtype=np.uint8
                        )
                    )

                # -------------------------------------------------
                # Create original display image
                # CHW -> HWC
                # -------------------------------------------------

                display_original = np.transpose(
                    rgb_normalized,
                    (1, 2, 0)
                )

                display_original = (
                    display_original * 255
                ).astype(np.uint8)


                # -------------------------------------------------
                # SECOND VIEW:
                # MASKS ONLY
                # -------------------------------------------------

                display_masks = display_original.copy()
                
                for mask in predicted_masks:

                    overlay = np.zeros_like(display_masks)

                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255)
                    )

                    overlay[mask == 1] = color

                    display_masks = cv2.addWeighted(
                        display_masks,
                        1.0,
                        overlay,
                        0.35,
                        0
                    )

                # -------------------------------------------------
                # THIRD VIEW:
                # MASKS + BOXES + CONFIDENCE SCORES
                # -------------------------------------------------

                display_prediction = display_masks.copy()

                for box, score in zip(
                    boxes,
                    scores
                ):

                    x1, y1, x2, y2 = (
                        box.astype(int)
                    )

                    cv2.rectangle(
                        display_prediction,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        1
                    )

                    cv2.putText(
                        display_prediction,
                        f"{score:.2f}",
                        (
                            x1,
                            max(y1 - 3, 10)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 255, 255),
                        1
                    )


                # -------------------------------------------------
                # Put all THREE images side-by-side
                # -------------------------------------------------

                combined = np.hstack(
                    (
                        display_original,
                        display_masks,
                        display_prediction
                    )
                )

                combined = cv2.cvtColor(
                    combined,
                    cv2.COLOR_RGB2BGR
                )


                # -------------------------------------------------
                # Labels
                # -------------------------------------------------

                image_width = display_original.shape[1]

                cv2.putText(
                    combined,
                    "ORIGINAL",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

                cv2.putText(
                    combined,
                    "MASKS",
                    (image_width + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

                cv2.putText(
                    combined,
                    f"MODEL >= {confidence_threshold:.2f}",
                    ((image_width * 2) + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
                # -------------------------------------------------
                # Enlarge review window
                # -------------------------------------------------

                combined_display = (
                    cv2.resize(
                        combined,
                        None,
                        fx=6,
                        fy=6,
                        interpolation=(
                            cv2.INTER_NEAREST
                        )
                    )
                )

                # -------------------------------------------------
                # Terminal information
                # -------------------------------------------------

                print(
                    f"\n{image_id} "
                    f"Observation "
                    f"{observation_index}"
                )

                print(
                    f"Predicted objects "
                    f"above "
                    f"{confidence_threshold:.2f}: "
                    f"{len(boxes)}"
                )

                if len(scores) > 0:

                    print(
                        f"Highest confidence: "
                        f"{scores.max():.3f}"
                    )

                    print(
                        f"Lowest accepted "
                        f"confidence: "
                        f"{scores.min():.3f}"
                    )

                print(
                    "A = Accept | "
                    "Z = Redraw | "
                    "C = Clear | "
                    "ESC = Exit"
                )

                cv2.imshow(
                    review_window,
                    combined_display
                )

                # -------------------------------------------------
                # Wait for human decision
                # -------------------------------------------------

                while True:

                    key = (
                        cv2.waitKey(0)
                        & 0xFF
                    )

                    # =============================================
                    # A = ACCEPT
                    # =============================================

                    if key == ord("a"):

                        print(
                            "Accepted model prediction."
                        )

                        boxes_file = (
                            annotation_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_boxes.npy"
                            )
                        )

                        labels_file = (
                            annotation_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_labels.npy"
                            )
                        )

                        masks_file = (
                            mask_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_masks.npy"
                            )
                        )

                        np.save(
                            boxes_file,
                            boxes.astype(
                                np.float32
                            )
                        )

                        np.save(
                            labels_file,
                            labels.astype(
                                np.int64
                            )
                        )

                        np.save(
                            masks_file,
                            predicted_masks.astype(
                                np.uint8
                            )
                        )

                        print(
                            f"Saved "
                            f"{len(labels)} "
                            f"instances."
                        )

                        break

                    # =============================================
                    # C = CLEAR
                    # =============================================

                    elif key == ord("c"):

                        print(
                            "Clearing observation."
                        )

                        empty_boxes = (
                            np.empty(
                                (0, 4),
                                dtype=np.float32
                            )
                        )

                        empty_labels = (
                            np.empty(
                                (0,),
                                dtype=np.int64
                            )
                        )

                        empty_masks = (
                            np.empty(
                                (
                                    0,
                                    height,
                                    width
                                ),
                                dtype=np.uint8
                            )
                        )

                        boxes_file = (
                            annotation_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_boxes.npy"
                            )
                        )

                        labels_file = (
                            annotation_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_labels.npy"
                            )
                        )

                        masks_file = (
                            mask_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_masks.npy"
                            )
                        )

                        np.save(
                            boxes_file,
                            empty_boxes
                        )

                        np.save(
                            labels_file,
                            empty_labels
                        )

                        np.save(
                            masks_file,
                            empty_masks
                        )

                        print(
                            "Saved empty annotation."
                        )

                        break

                    # =============================================
                    # Z = MANUAL REDRAW
                    # =============================================

                    elif key == ord("z"):

                        print(
                            "Manual redraw selected."
                        )

                        redraw_rgb = (
                            display_original
                            .astype(
                                np.float32
                            )
                            / 255.0
                        )

                        redraw_result = (
                            create_mask.draw_polygons(
                                rgb=redraw_rgb,
                                file_name=image_id,
                                observation_index=(
                                    observation_index
                                )
                            )
                        )

                        # User pressed ESC
                        if redraw_result is None:

                            print(
                                "Redraw cancelled."
                            )

                            continue

                        # Your existing method returns:
                        #
                        # polygons,
                        # annotation_save_path

                        polygons, _ = (
                            redraw_result
                        )

                        # -----------------------------------------
                        # Convert polygons into targets
                        # -----------------------------------------

                        (
                            boxes_redrawn,
                            labels_redrawn,
                            masks_redrawn
                        ) = (
                            create_mask
                            .create_training_targets(
                                polygons=polygons,
                                display_size=(
                                    850,
                                    850
                                ),
                                native_size=(
                                    width,
                                    height
                                )
                            )
                        )

                        boxes_file = (
                            annotation_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_boxes.npy"
                            )
                        )

                        labels_file = (
                            annotation_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_labels.npy"
                            )
                        )

                        masks_file = (
                            mask_path
                            / (
                                f"{image_id}_obs_"
                                f"{observation_index:03d}"
                                "_masks.npy"
                            )
                        )

                        np.save(
                            boxes_file,
                            boxes_redrawn.astype(
                                np.float32
                            )
                        
                        )

                        np.save(
                            labels_file,
                            labels_redrawn.astype(
                                np.int64
                            )
                        )

                        np.save(
                            masks_file,
                            masks_redrawn.astype(
                                np.uint8
                            )
                        )

                        print(
                            f"Redrew and saved "
                            f"{len(labels_redrawn)} "
                            f"instances."
                        )

                        break

                    # =============================================
                    # ESC = EXIT
                    # =============================================

                    elif key == 27:

                        cv2.destroyAllWindows()

                        print(
                            "Review stopped."
                        )

                        return

                # -------------------------------------------------
                # Close this observation
                # -------------------------------------------------

                try:

                    cv2.destroyWindow(
                        review_window
                    )

                except cv2.error:

                    pass

        # ---------------------------------------------------------
        # Finished directory
        # ---------------------------------------------------------

        cv2.destroyAllWindows()

        print(
            "\nFinished reviewing all observations."
        )

# =============================================================
# LOAD EXISTING MASK R-CNN
# =============================================================

def load_existing_model(
    model_path=r"C:\Python Stuff\Programming_Folder\mask_rcnn_epoch_10.pth"
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")

    # ---------------------------------------------------------
    # Recreate exact architecture used during training
    # ---------------------------------------------------------

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    num_classes = 2

    # ---------------------------------------------------------
    # Recreate box predictor
    # ---------------------------------------------------------

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            num_classes
        )
    )

    # ---------------------------------------------------------
    # Recreate mask predictor
    # ---------------------------------------------------------

    in_features_mask = (
        model.roi_heads
        .mask_predictor
        .conv5_mask
        .in_channels
    )

    model.roi_heads.mask_predictor = (
        MaskRCNNPredictor(
            in_features_mask,
            256,
            num_classes
        )
    )

    # ---------------------------------------------------------
    # Load trained weights
    # ---------------------------------------------------------

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")

    return model



if __name__ == "__main__":

    validator = NpyLoadFile(
        annotation_root=Path(
            r"D:\Portfolio\Custom_maks\Annotations"
        ),
        mask_root=Path(
            r"D:\Portfolio\Custom_maks\Masks"
        ),
        image_root=Path(
            r"D:\Portfolio\PASTIS-R\DATA_S2 - think I want these"

        ),
        root_dir=Path
        (r"D:\Portfolio\PASTIS-R\DATA_S2 - think I want these"
         ),
         file_path=Path
                 (r"D:\Portfolio\PASTIS-R\DATA_S2 - think I want these"
                  ), 
    )

    


    validator.validate_annotations()

