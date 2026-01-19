# common/colorimetry_analyzer.py
# ============================================================================
# NewFeature.md: Colorimetría Unificada (líneas 505-669)
# ============================================================================

from typing import Dict, List, Tuple, Optional
import numpy as np
from .colorimetry_config import (
    EYE_COLOR_RANGES,
    PALM_COLOR_RANGES,
    SKIN_TONE_CATEGORIES
)


class ColorimetryAnalyzer:
    """
    Analizador unificado de colorimetría para ojos, palmas y tono de piel.
    
    Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
    - Ojos: líneas 505-608
    - Palmas: líneas 609-669
    """
    
    def __init__(self):
        self.eye_ranges = EYE_COLOR_RANGES
        self.palm_ranges = PALM_COLOR_RANGES
        self.skin_categories = SKIN_TONE_CATEGORIES
    
    def analyze_eye_color(
        self, 
        rgb_avg: np.ndarray,
        rgb_dominant: Optional[List[Tuple[np.ndarray, float]]] = None,
        top_n: int = 2
    ) -> List[Tuple[str, float]]:
        """
        NewFeature.md línea 507: "Solo pasar los principales 2 resultados"
        
        Requisito línea 507:
        "Criterio inclusivo, si son 2 posibilidades se toman ambos diagnósticos al 50%"
        
        Args:
            rgb_avg: RGB promedio del iris
            rgb_dominant: Lista de colores dominantes (opcional)
            top_n: Número de resultados a retornar (default 2)
        
        Returns:
            Lista de (color_name, confidence) con top_n resultados
        """
        r, g, b = int(rgb_avg[0]), int(rgb_avg[1]), int(rgb_avg[2])
        matches = []
        
        for color_name, color_data in self.eye_ranges.items():
            # Check RGB ranges
            r_min, r_max = color_data['r_range']
            g_min, g_max = color_data['g_range']
            b_min, b_max = color_data['b_range']
            
            if not (r_min <= r <= r_max and 
                    g_min <= g <= g_max and 
                    b_min <= b <= b_max):
                continue
            
            # Check additional conditions
            condition = color_data.get('condition')
            if condition and not condition(r, g, b):
                continue
            
            matches.append(color_name)
        
        # NewFeature.md línea 507: Criterio inclusivo
        if len(matches) == 0:
            return []
        elif len(matches) == 1:
            return [(matches[0], 1.0)]
        elif len(matches) == 2:
            # Si son 2 posibilidades, ambos al 50%
            return [(matches[0], 0.5), (matches[1], 0.5)]
        else:
            # Si son más, distribuir equitativamente y tomar top_n
            confidence = 1.0 / len(matches)
            results = [(m, confidence) for m in matches]
            return results[:top_n]
    
    def analyze_palm_color(
        self,
        rgb_dominant: List[Tuple[np.ndarray, float]],
        top_n: int = 2
    ) -> List[Tuple[str, float]]:
        """
        NewFeature.md línea 611: "Solo pasar los principales 2 resultados"
        
        Requisito línea 611:
        "excluir average en diagnostico, solo tomar top 3 de colores dominantes"
        
        Args:
            rgb_dominant: Lista de (color_rgb, percentage) dominantes
            top_n: Número de resultados (default 2)
        
        Returns:
            Lista de (category, confidence) con top_n resultados
        """
        # NewFeature.md línea 611: Solo top 3 colores dominantes
        top_3_colors = rgb_dominant[:3]
        
        all_matches = []
        
        for color_rgb, percentage in top_3_colors:
            r, g, b = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])
            
            for category_name, category_data in self.palm_ranges.items():
                # Check RGB ranges
                r_min, r_max = category_data['r_range']
                g_min, g_max = category_data['g_range']
                b_min, b_max = category_data['b_range']
                
                if not (r_min <= r <= r_max and 
                        g_min <= g <= g_max and 
                        b_min <= b <= b_max):
                    continue
                
                # Check condition
                condition = category_data.get('condition')
                if condition and not condition(r, g, b):
                    continue
                
                all_matches.append((category_name, percentage / 100.0))
        
        # Consolidar y tomar top_n
        if not all_matches:
            return []
        
        # Agrupar por categoría y sumar confianzas
        consolidated = {}
        for cat, conf in all_matches:
            consolidated[cat] = consolidated.get(cat, 0) + conf
        
        # Ordenar y tomar top_n
        sorted_results = sorted(
            consolidated.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_results[:top_n]
    
    def analyze_skin_tone(
        self,
        face_rgb: Optional[np.ndarray] = None,
        palm_categories: Optional[List[Tuple[str, float]]] = None
    ) -> Dict:
        """
        Análisis de tono de piel combinado (facial + palmas).
        
        Categorías de NewFeature.md líneas 435-477:
        - rojiza/sanguineo
        - amarillenta/nervioso
        - cafe/oscura/bilioso
        - palida/blanca/linfatico
        
        Args:
            face_rgb: RGB promedio de región facial (opcional)
            palm_categories: Resultados de analyze_palm_color (opcional)
        
        Returns:
            Dict con tono dominante y confianza
        """
        skin_tone_votes = {}
        
        # Votar desde palmas
        if palm_categories:
            for palm_cat, confidence in palm_categories:
                for tone_name, palm_list in self.skin_categories.items():
                    if palm_cat in palm_list:
                        skin_tone_votes[tone_name] = skin_tone_votes.get(tone_name, 0) + confidence
        
        # Votar desde rostro (si disponible)
        if face_rgb is not None:
            # Analizar RGB facial para determinar tono
            r, g, b = int(face_rgb[0]), int(face_rgb[1]), int(face_rgb[2])
            
            # Heurísticas simples para tono facial
            if r > g * 1.15 and r > b * 1.15:
                skin_tone_votes['rojiza/sanguineo'] = skin_tone_votes.get('rojiza/sanguineo', 0) + 0.5
            elif abs(r - g) < 20 and (r + g) / 2 > b * 1.2:
                skin_tone_votes['amarillenta/nervioso'] = skin_tone_votes.get('amarillenta/nervioso', 0) + 0.5
            elif max(r, g, b) < 120:
                skin_tone_votes['cafe/oscura/bilioso'] = skin_tone_votes.get('cafe/oscura/bilioso', 0) + 0.5
            elif min(r, g, b) > 180:
                skin_tone_votes['palida/blanca/linfatico'] = skin_tone_votes.get('palida/blanca/linfatico', 0) + 0.5
        
        if not skin_tone_votes:
            return {'tone': None, 'confidence': 0.0}
        
        # Retornar tono dominante
        dominant_tone = max(skin_tone_votes.items(), key=lambda x: x[1])
        
        return {
            'tone': dominant_tone[0],
            'confidence': dominant_tone[1],
            'all_votes': skin_tone_votes
        }
