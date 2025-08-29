import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

class AgeVisualizationUtils:
    """
    Utility class for creating age estimation visualizations
    """
    
    @staticmethod
    def create_age_analysis_visualization(image: np.ndarray, 
                                        face_info: Dict[str, Any],
                                        age_info: Dict[str, Any],
                                        analysis_id: str,
                                        output_dir: str = "/app/results") -> str:
        """
        Create comprehensive age analysis visualization
        
        Args:
            image: Original image
            face_info: Face detection information
            age_info: Age estimation results
            analysis_id: Unique analysis identifier
            output_dir: Output directory for visualization
            
        Returns:
            Path to saved visualization
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Age Estimation Analysis Report', fontsize=16, fontweight='bold')
            
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Original image with face detection
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title('Face Detection', fontweight='bold')
            
            # Draw face bounding box
            bbox = face_info["bbox"]
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='lime', linewidth=3)
            axes[0, 0].add_patch(rect)
            
            # Add age text
            age_text = f"Age: {age_info['estimated_age']:.1f} years"
            axes[0, 0].text(x1, y1-10, age_text, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                           fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # 2. Face crop with detailed information
            face_crop = image_rgb[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else image_rgb
            axes[0, 1].imshow(face_crop)
            axes[0, 1].set_title('Extracted Face', fontweight='bold')
            axes[0, 1].axis('off')
            
            # Add detailed text information
            info_text = f"""Age: {age_info['estimated_age']:.1f} years
Category: {age_info['age_category'].replace('_', ' ').title()}
Confidence: {age_info['confidence']:.1%}
Range: {age_info['age_range']['min']}-{age_info['age_range']['max']} years
Orientation: {face_info['orientation'].title()}
Quality: {face_info['quality_assessment']['detection_quality'].title()}"""
            
            axes[0, 1].text(0.02, 0.98, info_text, 
                           transform=axes[0, 1].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                           fontsize=10)
            
            # 3. Age confidence visualization
            AgeVisualizationUtils._create_confidence_chart(axes[1, 0], age_info)
            
            # 4. Analysis metrics
            AgeVisualizationUtils._create_metrics_summary(axes[1, 1], face_info, age_info)
            
            plt.tight_layout()
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"age_analysis_{timestamp}_{analysis_id[:8]}.png"
            output_path = os.path.join(output_dir, filename)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Age analysis visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating age analysis visualization: {e}")
            return ""
    
    @staticmethod
    def _create_confidence_chart(ax, age_info: Dict[str, Any]):
        """Create confidence visualization chart"""
        try:
            confidence = age_info["confidence"]
            estimated_age = age_info["estimated_age"]
            age_range = age_info["age_range"]
            
            # Create confidence bar
            categories = ['Age Estimation\nConfidence']
            values = [confidence]
            colors = ['green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Confidence Score')
            ax.set_title('Estimation Confidence', fontweight='bold')
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
            
            # Add age range information
            range_text = f"Estimated Age: {estimated_age:.1f} years\nRange: {age_range['min']}-{age_range['max']} years"
            ax.text(0.5, 0.3, range_text, transform=ax.transAxes,
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
        except Exception as e:
            logger.error(f"Error creating confidence chart: {e}")
            ax.text(0.5, 0.5, 'Confidence chart\nunavailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _create_metrics_summary(ax, face_info: Dict[str, Any], age_info: Dict[str, Any]):
        """Create metrics summary visualization"""
        try:
            ax.axis('off')
            ax.set_title('Analysis Metrics', fontweight='bold')
            
            # Prepare metrics data
            metrics = [
                ("Detection Quality", face_info["quality_assessment"]["detection_quality"]),
                ("Size Adequacy", face_info["quality_assessment"]["size_adequacy"]),
                ("Face Orientation", face_info["orientation"]),
                ("Age Reliability", age_info["reliability"]),
                ("Age Category", age_info["age_category"].replace('_', ' ')),
                ("Coverage Ratio", f"{face_info['image_coverage_ratio']:.1%}")
            ]
            
            # Create text summary
            y_pos = 0.9
            for metric, value in metrics:
                # Color coding for quality metrics
                if metric in ["Detection Quality", "Size Adequacy", "Age Reliability"]:
                    if value in ["excellent", "high"]:
                        color = "green"
                    elif value in ["good", "medium"]:
                        color = "orange"
                    else:
                        color = "red"
                else:
                    color = "black"
                
                ax.text(0.05, y_pos, f"{metric}:", fontweight='bold', 
                       transform=ax.transAxes)
                ax.text(0.55, y_pos, str(value).title(), color=color, 
                       transform=ax.transAxes, fontweight='bold')
                y_pos -= 0.12
            
            # Add face dimensions
            face_dims = face_info["face_dimensions"]
            dimensions_text = f"Face Size: {face_dims['width']}x{face_dims['height']} px"
            ax.text(0.05, 0.15, dimensions_text, transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
            
        except Exception as e:
            logger.error(f"Error creating metrics summary: {e}")
            ax.text(0.5, 0.5, 'Metrics summary\nunavailable', 
                   ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def create_batch_analysis_chart(batch_results: Dict[str, Any],
                                  output_dir: str = "/app/results") -> str:
        """
        Create batch analysis summary chart
        
        Args:
            batch_results: Batch processing results
            output_dir: Output directory
            
        Returns:
            Path to saved chart
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle('Batch Age Estimation Analysis', fontsize=16, fontweight='bold')
            
            successful_results = [r for r in batch_results["individual_results"] if r["success"]]
            
            if not successful_results:
                # No successful results
                fig.text(0.5, 0.5, 'No successful age estimations in batch', 
                        ha='center', va='center', fontsize=16)
                plt.tight_layout()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_analysis_{timestamp}_{batch_results['batch_id'][:8]}.png"
                output_path = os.path.join(output_dir, filename)
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                return output_path
            
            # 1. Age distribution histogram
            ages = [r["age_estimation"]["estimated_age"] for r in successful_results]
            axes[0, 0].hist(ages, bins=min(10, len(set(ages))), alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Estimated Age (years)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Age Distribution')
            axes[0, 0].axvline(np.mean(ages), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(ages):.1f}')
            axes[0, 0].legend()
            
            # 2. Age categories pie chart
            categories = batch_results["batch_summary"]["age_categories"]
            if categories:
                axes[0, 1].pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%',
                              startangle=90)
                axes[0, 1].set_title('Age Categories Distribution')
            else:
                axes[0, 1].text(0.5, 0.5, 'No category data', ha='center', va='center')
                axes[0, 1].set_title('Age Categories Distribution')
            
            # 3. Confidence scores
            confidences = [r["age_estimation"]["confidence"] for r in successful_results]
            axes[1, 0].hist(confidences, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Confidence Score Distribution')
            axes[1, 0].axvline(np.mean(confidences), color='red', linestyle='--',
                              label=f'Mean: {np.mean(confidences):.2f}')
            axes[1, 0].legend()
            
            # 4. Processing statistics
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Batch Statistics', fontweight='bold')
            
            stats_text = f"""Total Images: {batch_results['total_images']}
Successful: {batch_results['successful_analyses']}
Success Rate: {batch_results['success_rate']:.1%}
Total Time: {batch_results['total_processing_time']:.2f}s
Avg Time/Image: {batch_results['average_processing_time']:.2f}s
Age Range: {min(ages):.1f} - {max(ages):.1f} years
Average Age: {batch_results['batch_summary']['average_age']:.1f} years"""
            
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_analysis_{timestamp}_{batch_results['batch_id'][:8]}.png"
            output_path = os.path.join(output_dir, filename)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Batch analysis chart saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating batch analysis chart: {e}")
            return ""
    
    @staticmethod
    def create_simple_overlay(image: np.ndarray, 
                            age_info: Dict[str, Any],
                            face_bbox: List[int]) -> np.ndarray:
        """
        Create simple age overlay on image
        
        Args:
            image: Input image
            age_info: Age estimation information
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Image with age overlay
        """
        try:
            overlay_image = image.copy()
            x1, y1, x2, y2 = face_bbox
            
            # Draw face rectangle
            cv2.rectangle(overlay_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add age text
            age = age_info["estimated_age"]
            confidence = age_info["confidence"]
            text = f"Age: {age:.0f} ({confidence:.0%})"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw text background
            text_x = x1
            text_y = max(y1 - 10, text_size[1] + 10)
            cv2.rectangle(overlay_image, 
                         (text_x, text_y - text_size[1] - 5),
                         (text_x + text_size[0] + 10, text_y + 5),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(overlay_image, text, (text_x + 5, text_y),
                       font, font_scale, (255, 255, 255), thickness)
            
            return overlay_image
            
        except Exception as e:
            logger.error(f"Error creating simple overlay: {e}")
            return image