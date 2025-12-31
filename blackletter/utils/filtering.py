"""Box filtering and deduplication utilities."""

from typing import List, Dict


class BoxFilter:
    """Filter and manage bounding boxes."""

    @staticmethod
    def filter_overlapping_boxes(
            objects: List[Dict], overlap_threshold: float = 0.6
    ) -> List[Dict]:
        """Remove overlapping boxes on same page based on priority rules.

        Keeps higher-confidence or larger boxes depending on label type.

        Args:
            objects: List of detection dicts with 'page_index', 'label', 'coords'
            overlap_threshold: IoU threshold for considering boxes overlapping

        Returns:
            Filtered list without overlapping boxes
        """
        cleaned_objects = []
        pages = sorted(list(set(o["page_index"] for o in objects)))

        for page_idx in pages:
            page_objs = [o for o in objects if o["page_index"] == page_idx]
            indices_to_remove = set()

            for i in range(len(page_objs)):
                for j in range(i + 1, len(page_objs)):
                    if i in indices_to_remove or j in indices_to_remove:
                        continue

                    obj_a = page_objs[i]
                    obj_b = page_objs[j]

                    # Only compare same label types
                    if obj_a["label"] != obj_b["label"]:
                        continue

                    overlap = BoxFilter.calculate_iou(
                        obj_a["coords"], obj_b["coords"]
                    )

                    if overlap > overlap_threshold:
                        label = obj_a["label"]
                        area_a = BoxFilter.rect_area(obj_a["coords"])
                        area_b = BoxFilter.rect_area(obj_b["coords"])

                        # For captions, keep the largest
                        if label == "caption":
                            indices_to_remove.add(j if area_a > area_b else i)
                        else:
                            # For others, keep the largest
                            indices_to_remove.add(i if area_a < area_b else j)

            for idx, obj in enumerate(page_objs):
                if idx not in indices_to_remove:
                    cleaned_objects.append(obj)

        return cleaned_objects

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union for two boxes.

        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Intersection
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def rect_area(box: List[float]) -> float:
        """Calculate rectangle area.

        Args:
            box: [x1, y1, x2, y2] format

        Returns:
            Area in square units
        """
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
