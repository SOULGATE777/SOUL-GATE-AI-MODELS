import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ProfileRotationVisualizer:
    """
    Visualization utilities for profile rotation analysis results
    """
    
    def __init__(self):
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color scheme for different rotation categories
        self.colors = {
            'aceptable': '#2E8B57',  # Sea Green
            'hacia_abajo': '#FF6B6B',  # Light Red
            'hacia_arriba': '#FF8E53',  # Orange
            'desde_abajo_o_mandibula_hacia_primer_plano': '#FFD93D',  # Yellow
            'desde_arriba_o_superior_hacia_primer_plano': '#A8E6CF',  # Light Green
            'rotado_lejos_o_desde_atras': '#FF6B9D',  # Pink
            'rotado_hacia_desde_adelante': '#C7CEEA',  # Light Purple
            'default': '#808080'  # Gray
        }
        
        # Status colors
        self.status_colors = {
            'suitable': '#28a745',  # Green
            'unsuitable': '#dc3545',  # Red
            'uncertain': '#ffc107'  # Yellow
        }
    
    def create_rotation_analysis_visualization(
        self, 
        image: np.ndarray, 
        results: Dict[str, Any], 
        save_path: str
    ) -> bool:
        """
        Create comprehensive visualization of rotation analysis results
        
        Args:
            image: Original input image
            results: Analysis results from ProfileRotationPipeline
            save_path: Path to save the visualization
            
        Returns:
            True if visualization created successfully
        """
        try:
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Profile Rotation Analysis Results', fontsize=16, fontweight='bold')
            
            # 1. Original image with analysis status
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Profile Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Add status overlay
            rotation_assessment = results.get('rotation_assessment', {})
            is_suitable = rotation_assessment.get('is_suitable', False)
            
            status_text = "✅ SUITABLE" if is_suitable else "❌ NOT SUITABLE"
            status_color = self.status_colors['suitable'] if is_suitable else self.status_colors['unsuitable']
            
            ax1.text(0.02, 0.98, status_text, transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold', color=status_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 2. Confidence scores visualization
            confidences = rotation_assessment.get('all_probabilities', {})
            if confidences:
                self._plot_confidence_scores(ax2, confidences, results)
            else:
                ax2.text(0.5, 0.5, 'No confidence data available', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Confidence Scores')
            
            # 3. Rotation assessment details
            self._plot_rotation_assessment(ax3, results)
            
            # 4. Recommendations and summary
            self._plot_recommendations(ax4, results)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Rotation analysis visualization saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating rotation analysis visualization: {e}")
            return False
    
    def _plot_confidence_scores(self, ax, confidences: Dict[str, float], results: Dict[str, Any]):
        """Plot confidence scores for all rotation classes"""
        class_names = list(confidences.keys())
        confidence_values = list(confidences.values())
        
        # Create horizontal bar chart
        y_pos = np.arange(len(class_names))
        
        # Color bars based on prediction status
        rotation_assessment = results.get('rotation_assessment', {})
        predicted_tags = rotation_assessment.get('predicted_tags', [])
        threshold = rotation_assessment.get('threshold_used', 0.5)
        
        colors = []
        for class_name in class_names:
            if class_name in predicted_tags:
                colors.append(self.colors.get(class_name, self.colors['default']))
            else:
                colors.append('#E0E0E0')  # Light gray for non-predicted
        
        bars = ax.barh(y_pos, confidence_values, color=colors, alpha=0.8)
        
        # Add threshold line
        ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Threshold ({threshold:.2f})')
        
        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace('_', ' ').title() for name in class_names])
        ax.set_xlabel('Confidence Score')
        ax.set_title('Class Confidence Scores', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, confidence_values)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=9)
    
    def _plot_rotation_assessment(self, ax, results: Dict[str, Any]):
        """Plot detailed rotation assessment information"""
        rotation_assessment = results.get('rotation_assessment', {})
        viability = results.get('viability_for_analysis', {})
        
        # Clear axis
        ax.clear()
        ax.axis('off')
        
        # Assessment details
        predicted_tags = rotation_assessment.get('predicted_tags', [])
        rotation_issues = rotation_assessment.get('rotation_issues', [])
        is_suitable = rotation_assessment.get('is_suitable', False)
        certainty = rotation_assessment.get('prediction_certainty', 'unknown')
        max_confidence = rotation_assessment.get('max_confidence', 0.0)
        
        # Create text summary
        y_pos = 0.9
        line_height = 0.08
        
        # Title
        ax.text(0.05, y_pos, 'Rotation Assessment Details', 
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height * 1.5
        
        # Suitability
        suitability_text = "✅ Suitable for Analysis" if is_suitable else "❌ Not Suitable for Analysis"
        suitability_color = self.status_colors['suitable'] if is_suitable else self.status_colors['unsuitable']
        ax.text(0.05, y_pos, suitability_text, fontsize=12, fontweight='bold',
               color=suitability_color, transform=ax.transAxes)
        y_pos -= line_height
        
        # Predicted orientation
        if predicted_tags:
            orientation_text = f"Predicted Orientation: {', '.join(predicted_tags).replace('_', ' ').title()}"
        else:
            orientation_text = "Predicted Orientation: Unclear"
        ax.text(0.05, y_pos, orientation_text, fontsize=11, transform=ax.transAxes)
        y_pos -= line_height
        
        # Rotation issues
        if rotation_issues:
            issues_text = f"Rotation Issues: {', '.join(rotation_issues).replace('_', ' ').title()}"
            ax.text(0.05, y_pos, issues_text, fontsize=11, 
                   color=self.status_colors['unsuitable'], transform=ax.transAxes)
            y_pos -= line_height
        
        # Confidence metrics
        ax.text(0.05, y_pos, f"Maximum Confidence: {max_confidence:.3f}", 
               fontsize=11, transform=ax.transAxes)
        y_pos -= line_height
        
        ax.text(0.05, y_pos, f"Prediction Certainty: {certainty.title()}", 
               fontsize=11, transform=ax.transAxes)
        y_pos -= line_height
        
        # Analysis suitability
        y_pos -= line_height * 0.5
        ax.text(0.05, y_pos, 'Analysis Suitability:', 
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height
        
        anthro_suitable = viability.get('suitable_for_anthropometric', False)
        morpho_suitable = viability.get('suitable_for_morphological', False)
        
        anthro_text = f"  Anthropometric: {'✅ Yes' if anthro_suitable else '❌ No'}"
        morpho_text = f"  Morphological: {'✅ Yes' if morpho_suitable else '❌ No'}"
        
        ax.text(0.05, y_pos, anthro_text, fontsize=11, transform=ax.transAxes)
        y_pos -= line_height
        ax.text(0.05, y_pos, morpho_text, fontsize=11, transform=ax.transAxes)
        
        # Add border
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, 
                              linewidth=1, edgecolor='gray', 
                              facecolor='none', transform=ax.transAxes))
    
    def _plot_recommendations(self, ax, results: Dict[str, Any]):
        """Plot recommendations and summary information"""
        # Clear axis
        ax.clear()
        ax.axis('off')
        
        viability = results.get('viability_for_analysis', {})
        analysis_summary = results.get('analysis_summary', {})
        
        # Title
        y_pos = 0.9
        line_height = 0.08
        
        ax.text(0.05, y_pos, 'Recommendations & Summary', 
               fontsize=14, fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height * 1.5
        
        # Main finding
        main_finding = analysis_summary.get('main_finding', 'No summary available')
        ax.text(0.05, y_pos, f"Main Finding:", fontsize=12, fontweight='bold', 
               transform=ax.transAxes)
        y_pos -= line_height
        
        # Wrap long text
        wrapped_finding = self._wrap_text(main_finding, 50)
        for line in wrapped_finding:
            ax.text(0.05, y_pos, line, fontsize=11, transform=ax.transAxes)
            y_pos -= line_height * 0.8
        
        y_pos -= line_height * 0.5
        
        # Recommendations
        recommendation = viability.get('recommendation', 'No specific recommendations')
        ax.text(0.05, y_pos, f"Recommendation:", fontsize=12, fontweight='bold', 
               transform=ax.transAxes)
        y_pos -= line_height
        
        wrapped_rec = self._wrap_text(recommendation, 50)
        for line in wrapped_rec:
            ax.text(0.05, y_pos, line, fontsize=11, transform=ax.transAxes)
            y_pos -= line_height * 0.8
        
        # Analysis info
        y_pos -= line_height * 0.5
        analysis_id = results.get('analysis_id', 'Unknown')
        ax.text(0.05, y_pos, f"Analysis ID: {analysis_id}", 
               fontsize=10, style='italic', transform=ax.transAxes)
        
        # Add border
        ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, 
                              linewidth=1, edgecolor='gray', 
                              facecolor='none', transform=ax.transAxes))
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def create_simple_visualization(
        self, 
        image: np.ndarray, 
        predicted_tags: List[str], 
        is_suitable: bool, 
        confidence: float,
        save_path: str
    ) -> bool:
        """
        Create simple visualization for basic rotation classification
        
        Args:
            image: Original input image
            predicted_tags: List of predicted rotation tags
            is_suitable: Whether the profile is suitable for analysis
            confidence: Maximum confidence score
            save_path: Path to save the visualization
            
        Returns:
            True if visualization created successfully
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle('Profile Rotation Classification', fontsize=16, fontweight='bold')
            
            # Original image
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Profile Image', fontsize=14)
            ax1.axis('off')
            
            # Status overlay
            status_text = "✅ SUITABLE" if is_suitable else "❌ NOT SUITABLE"
            status_color = self.status_colors['suitable'] if is_suitable else self.status_colors['unsuitable']
            
            ax1.text(0.02, 0.98, status_text, transform=ax1.transAxes, 
                    fontsize=12, fontweight='bold', color=status_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Results summary
            ax2.axis('off')
            y_pos = 0.8
            
            ax2.text(0.1, y_pos, 'Classification Results:', 
                    fontsize=14, fontweight='bold', transform=ax2.transAxes)
            y_pos -= 0.1
            
            # Predicted tags
            tags_text = ', '.join(predicted_tags).replace('_', ' ').title() if predicted_tags else 'No clear orientation'
            ax2.text(0.1, y_pos, f"Orientation: {tags_text}", 
                    fontsize=12, transform=ax2.transAxes)
            y_pos -= 0.08
            
            # Confidence
            ax2.text(0.1, y_pos, f"Confidence: {confidence:.3f}", 
                    fontsize=12, transform=ax2.transAxes)
            y_pos -= 0.08
            
            # Suitability
            suitability_text = "Suitable for analysis" if is_suitable else "Not suitable for analysis"
            ax2.text(0.1, y_pos, f"Status: {suitability_text}", 
                    fontsize=12, color=status_color, transform=ax2.transAxes)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logger.info(f"Simple rotation visualization saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating simple visualization: {e}")
            return False