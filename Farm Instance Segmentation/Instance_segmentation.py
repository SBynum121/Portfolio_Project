import re

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2
import math


class NpyLoadFile:
    def __init__(self, file_path: Path, root_dir: Path):
        self.file_path = Path(file_path)
        self.root_dir = Path(root_dir)

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
    def single_obv_at_attime(self):
        annotation_path = Path(
            r"D:\Portfolio\Custom_maks\Annotations"
        )

        mask_path = Path(
            r"D:\Portfolio\Custom_maks\Masks"
        )

        annotation_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        for file_path in self.root_dir.iterdir():

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() != ".npy":
                continue

            img_array = np.load(file_path)
            number_of_images = img_array.shape[0]

            for observation_index in range(number_of_images):
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
                    / (channel_max - channel_min + 1e-8)
                )

                rgb = np.transpose(rgb, (1, 2, 0))
                rgb = np.clip(rgb, 0, 1)

                display_rgb = cv2.resize(
                    rgb,
                    (300, 300),
                    interpolation=cv2.INTER_LANCZOS4
                )

                polygons = create_mask.draw_polygons(
                    rgb=display_rgb,
                    file_name=file_path.name,
                    observation_index=observation_index
                )

                boxes, labels, masks = (
                    create_mask.create_training_targets(
                        polygons=polygons,
                        display_size=(300, 300),
                        native_size=(128, 128)
                    )
                )

                # Validate the arrays before saving.
                assert boxes.shape == (len(labels), 4)
                assert masks.shape == (
                    len(labels),
                    128,
                    128
                )

                assert boxes.dtype == np.float32
                assert labels.dtype == np.int64
                assert masks.dtype == np.uint8

                assert np.all(
                    (masks == 0) | (masks == 1)
                )

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
                    f"Saved {annotation_name}: "
                    f"{len(labels)} parcel instances"
                )
class create_mask():
    def __init__(self, mask_path:Path):
        self.mask_path = mask_path
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

            # Mouse logic here...

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

                break

            elif key == 27:
                # Escape cancels the current observation.
                polygons.clear()
                break

        cv2.destroyWindow(window_name)

        return polygons
    
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
My_dir = Path(r"D:\Portfolio\PASTIS-R\DATA_S2 - think I want these")
single_img = Path(r"D:\Portfolio\PASTIS-R\DATA_S2 - think I want these\S2_10003.npy")
if __name__ == "__main__":
    NPY = NpyLoadFile(file_path=single_img, root_dir=Path(My_dir))
    NPY.single_obv_at_attime()