import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional
import base64
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class FrontalVisualizationManager:
    """Visualization utilities for frontal preprocessing service"""

    def __init__(self):
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
        ]

    def get_color(self, index: int) -> Tuple[int, int, int]:
        """Get color for visualization by index"""
        return self.colors[index % len(self.colors)]

    def draw_bounding_boxes(self, image: np.ndarray,
                           detections: List[Dict],
                           thickness: int = 2,
                           font_scale: float = 0.6) -> np.ndarray:
        """
        Draw bounding boxes on image with detection information

        Args:
            image: Input image in RGB format
            detections: List of detection dictionaries
            thickness: Line thickness for bounding boxes
            font_scale: Font scale for text

        Returns:
            Image with drawn bounding boxes
        """
        vis_image = image.copy()

        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection.get('class_name', 'head')

            # Get color for this detection
            color = self.get_color(i)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Draw background rectangle for text
            cv2.rectangle(vis_image,
                         (x1, y1 - text_height - baseline - 5),
                         (x1 + text_width, y1),
                         color, -1)

            # Draw text
            cv2.putText(vis_image, label, (x1, y1 - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return vis_image

    def create_detection_grid(self, original_image: np.ndarray,
                             cropped_heads: List[np.ndarray],
                             detections: List[Dict],
                             grid_cols: int = 3) -> np.ndarray:
        """
        Create a grid visualization showing original image and cropped heads

        Args:
            original_image: Original input image
            cropped_heads: List of cropped head images
            detections: List of detection information
            grid_cols: Number of columns in the grid

        Returns:
            Grid visualization image
        """
        num_heads = len(cropped_heads)
        if num_heads == 0:
            return original_image

        # Calculate grid dimensions
        grid_rows = (num_heads + grid_cols - 1) // grid_cols + 1  # +1 for original image

        # Determine cell size based on original image
        cell_height = max(200, original_image.shape[0] // 3)
        cell_width = max(200, original_image.shape[1] // 3)

        # Create grid canvas
        grid_height = grid_rows * cell_height
        grid_width = grid_cols * cell_width
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Resize and place original image with bounding boxes
        original_with_boxes = self.draw_bounding_boxes(original_image, detections)
        original_resized = cv2.resize(original_with_boxes, (cell_width, cell_height))
        grid_image[0:cell_height, 0:cell_width] = original_resized

        # Add title to original image
        cv2.putText(grid_image, "Original + Detections", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Place cropped heads
        for i, (cropped_head, detection) in enumerate(zip(cropped_heads, detections)):
            row = (i + 1) // grid_cols + 1  # +1 to skip original image row
            col = (i + 1) % grid_cols

            if row >= grid_rows:
                break

            # Resize cropped head to cell size
            head_resized = cv2.resize(cropped_head, (cell_width, cell_height))

            # Calculate position in grid
            start_row = row * cell_height
            end_row = start_row + cell_height
            start_col = col * cell_width
            end_col = start_col + cell_width

            # Place in grid
            grid_image[start_row:end_row, start_col:end_col] = head_resized

            # Add title
            title = f"Head {i+1} (conf: {detection['confidence']:.2f})"
            cv2.putText(grid_image, title, (start_col + 10, start_row + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return grid_image

    def create_processing_summary(self, processing_result: Dict) -> np.ndarray:
        """
        Create a visual summary of processing results

        Args:
            processing_result: Result from pipeline processing

        Returns:
            Summary visualization image
        """
        # Create a white canvas
        canvas_width = 800
        canvas_height = 400
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

        # Title
        title = "Frontal Preprocessing Summary"
        cv2.putText(canvas, title, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # Statistics
        stats = [
            f"Total Detections: {processing_result['total_detections']}",
            f"Processed Heads: {len(processing_result['processed_heads'])}",
            f"Original Size: {processing_result['original_image_size']}",
        ]

        # Processing parameters
        params = processing_result['processing_parameters']
        param_stats = [
            f"Confidence Threshold: {params['confidence_threshold']:.2f}",
            f"Target Size: {params['target_size']}",
            f"Padding Factor: {params['padding_factor']:.2f}",
            f"Output Format: {params['output_format']}",
        ]

        # Draw statistics
        y = 80
        for stat in stats:
            cv2.putText(canvas, stat, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y += 30

        # Draw parameters
        y += 20
        cv2.putText(canvas, "Processing Parameters:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 30

        for param in param_stats:
            cv2.putText(canvas, param, (40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y += 25

        # Draw head details if available
        if processing_result['processed_heads']:
            y += 20
            cv2.putText(canvas, "Head Details:", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y += 30

            for i, head in enumerate(processing_result['processed_heads']):
                head_info = f"Head {i+1}: {head['class_name']}, conf: {head['confidence']:.3f}"
                cv2.putText(canvas, head_info, (40, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y += 25

                if y > canvas_height - 30:  # Prevent overflow
                    break

        return canvas

    def save_visualization_base64(self, image: np.ndarray,
                                 format: str = 'PNG',
                                 quality: int = 95) -> str:
        """
        Convert visualization to base64 string

        Args:
            image: Visualization image in RGB format
            format: Output format ('JPEG', 'PNG')
            quality: JPEG quality (1-100)

        Returns:
            Base64 encoded image string
        """
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))

            # Save to bytes buffer
            buffer = io.BytesIO()
            if format.upper() == 'JPEG':
                pil_image.save(buffer, format='JPEG', quality=quality)
            else:
                pil_image.save(buffer, format='PNG')

            # Encode to base64
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            return img_base64

        except Exception as e:
            logger.error(f"Error creating visualization base64: {str(e)}")
            raise e

    def create_complete_visualization(self, original_image: np.ndarray,
                                    processing_result: Dict,
                                    include_summary: bool = True) -> Dict:
        """
        Create complete visualization package

        Args:
            original_image: Original input image
            processing_result: Result from pipeline processing
            include_summary: Whether to include processing summary

        Returns:
            Dictionary with visualization data
        """
        visualizations = {}

        try:
            # Extract cropped heads from base64
            cropped_heads = []
            for head_data in processing_result['processed_heads']:
                # Decode base64 to get cropped head
                head_base64 = head_data['cropped_image_base64']
                head_bytes = base64.b64decode(head_base64)
                pil_image = Image.open(io.BytesIO(head_bytes))
                head_array = np.array(pil_image.convert('RGB'))
                cropped_heads.append(head_array)

            # Create detection grid
            if cropped_heads:
                detection_grid = self.create_detection_grid(
                    original_image,
                    cropped_heads,
                    processing_result['processed_heads']
                )
                visualizations['detection_grid'] = self.save_visualization_base64(detection_grid)

            # Create original with bounding boxes
            original_with_boxes = self.draw_bounding_boxes(
                original_image,
                processing_result['processed_heads']
            )
            visualizations['original_with_detections'] = self.save_visualization_base64(original_with_boxes)

            # Create processing summary
            if include_summary:
                summary_vis = self.create_processing_summary(processing_result)
                visualizations['processing_summary'] = self.save_visualization_base64(summary_vis)

            return {
                'status': 'success',
                'visualizations': visualizations,
                'total_visualizations': len(visualizations)
            }

        except Exception as e:
            logger.error(f"Error creating complete visualization: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'visualizations': {},
                'total_visualizations': 0
            }

    def create_side_by_side_comparison(self, original_image: np.ndarray,
                                     processed_heads: List[np.ndarray],
                                     titles: Optional[List[str]] = None) -> np.ndarray:
        """
        Create side-by-side comparison of original and processed images

        Args:
            original_image: Original input image
            processed_heads: List of processed head images
            titles: Optional titles for each image

        Returns:
            Side-by-side comparison image
        """
        if not processed_heads:
            return original_image

        # Determine dimensions
        num_images = len(processed_heads) + 1  # +1 for original
        canvas_height = max(original_image.shape[0], 400)
        canvas_width = num_images * max(original_image.shape[1] // num_images, 300)

        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Calculate cell width
        cell_width = canvas_width // num_images

        # Place original image
        original_resized = cv2.resize(original_image, (cell_width, canvas_height))
        canvas[:, 0:cell_width] = original_resized

        # Add title for original
        title = titles[0] if titles and len(titles) > 0 else "Original"
        cv2.putText(canvas, title, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Place processed heads
        for i, head in enumerate(processed_heads):
            start_col = (i + 1) * cell_width
            end_col = start_col + cell_width

            # Resize head to fit cell
            head_resized = cv2.resize(head, (cell_width, canvas_height))
            canvas[:, start_col:end_col] = head_resized

            # Add title
            title = titles[i + 1] if titles and len(titles) > i + 1 else f"Head {i + 1}"
            cv2.putText(canvas, title, (start_col + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return canvas