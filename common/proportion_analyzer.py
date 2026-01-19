# ============================================================================
# NewFeature.md: Proportion Analyzer - Lógica de proporción facial/frente (líneas 29-69)
# ============================================================================
# Analiza proporciones faciales y de frente para realizar splitting de diagnósticos.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
#
# Proporciones analizadas:
# - Proporción Facial (Derecha e Izquierda): Umbrales 0.99, 1.17
# - Proporción Frente (Derecha e Izquierda): Umbral 0.35
# ============================================================================

from typing import Tuple, Optional, Dict


class ProportionAnalyzer:
    """
    NewFeature.md líneas 29-69: Analizador de proporciones faciales y de frente.
    
    Implementa la lógica de splitting de diagnósticos basada en proporciones,
    donde un mismo diagnóstico base puede tener diferentes narrativas según
    la proporción medida.
    """
    
    def __init__(self):
        """Initialize proportion analyzer"""
        pass
    
    # ========================================================================
    # NewFeature.md: Proporción Facial (líneas 29-54)
    # ========================================================================
    
    def analyze_facial_proportion(
        self, 
        diagnosis: str, 
        proportion: float,
        confidence: float
    ) -> Tuple[str, str, Dict]:
        """
        NewFeature.md líneas 29-54: Splitting de diagnósticos por Proporción Facial.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 29-54
        
        Umbrales de proporción:
        - < 0.99 (línea 39)
        - >= 0.99 (línea 35)
        - >= 1.17 (líneas 33, 41, 43, etc.)
        
        Args:
            diagnosis: Diagnóstico base (e.g., 'sol_neptuno_combined')
            proportion: Proporción Facial (Derecha e Izquierda)
            confidence: Nivel de confianza del diagnóstico
            
        Returns:
            Tuple[adjusted_diagnosis, narrative_key, metadata]
            
        Example:
            # NewFeature.md línea 33-34: sol_neptuno con proporción < 1.17
            diag, narrative, meta = analyzer.analyze_facial_proportion(
                'sol_neptuno_combined', 1.05, 0.85
            )
            # diag = 'sol_neptuno_liderazgo'
            # narrative = 'liderazgo_carismatico_distante'
        """
        metadata = {
            'original_diagnosis': diagnosis,
            'proportion': proportion,
            'confidence': confidence,
            'proportion_thresholds': []
        }
        
        # ====================================================================
        # NewFeature.md líneas 33-34: sol_neptuno_combined
        # ====================================================================
        # "sol_neptuno_combined y proporción Facial y proporcion menor a 1.17:
        #  Estas personas necesitan mucho espacio y tiempo para sí mismas..."
        
        if diagnosis == 'sol_neptuno_combined':
            if proportion < 1.17:
                # NewFeature.md línea 33: proporción menor a 1.17
                metadata['proportion_thresholds'].append('< 1.17')
                return (
                    'sol_neptuno_liderazgo',
                    'liderazgo_carismatico_distante',
                    metadata
                )
        
        # ====================================================================
        # NewFeature.md líneas 35-36: marte_tierra_rectangulo con proporción > 0.99
        # ====================================================================
        # "marte_tierra_rectangulo y proporción facial mayor a .99:
        #  Muy orientados a la acción, físicos y probablemente impacientes..."
        
        elif diagnosis == 'marte_tierra_rectangulo':
            if proportion > 0.99:
                # NewFeature.md línea 35: proporción mayor a 0.99
                metadata['proportion_thresholds'].append('> 0.99')
                return (
                    'marte_tierra_accion',
                    'accion_fisica_impaciente',
                    metadata
                )
            else:
                # NewFeature.md línea 39: proporción igual o menor a 0.99
                metadata['proportion_thresholds'].append('<= 0.99')
                return (
                    'marte_tierra_concreto',
                    'concreto_practico_logico',
                    metadata
                )
        
        # ====================================================================
        # NewFeature.md líneas 41-42: saturno_trapezoide_base_angosta <= 1.17
        # ====================================================================
        # "Saturno_trapezoide_base_angosta proporción facial igual o menor a 1.17:
        #  Saben con claridad qué quieren lograr..."
        
        elif diagnosis == 'saturno_trapezoide_base_angosta':
            if proportion <= 1.17:
                # NewFeature.md línea 41: proporción igual o menor a 1.17
                metadata['proportion_thresholds'].append('<= 1.17')
                return (
                    'saturno_contemplativo',
                    'contemplativo_perseverante',
                    metadata
                )
            else:
                # NewFeature.md línea 43: proporción igual o mayor a 1.17
                metadata['proportion_thresholds'].append('>= 1.17')
                return (
                    'saturno_excentrico',
                    'excentrico_visionario',
                    metadata
                )
        
        # ====================================================================
        # NewFeature.md líneas 45-46: luna_jupiter_combined >= 1.17
        # ====================================================================
        # "luna_jupiter_combined y proporción facial igual o mayor a 1.17:
        #  Estables, decididos y profundamente leales..."
        
        elif diagnosis == 'luna_jupiter_combined':
            if proportion >= 1.17:
                # NewFeature.md línea 45: proporción igual o mayor a 1.17
                metadata['proportion_thresholds'].append('>= 1.17')
                return (
                    'jupiter_maestro',
                    'maestro_lider_justicia',
                    metadata
                )
            elif proportion < 0.99:
                # NewFeature.md línea 47: proporción menor a 0.99
                metadata['proportion_thresholds'].append('< 0.99')
                return (
                    'luna_sensible',
                    'sensible_imaginativo_poetico',
                    metadata
                )
            else:
                # Entre 0.99 y 1.17
                metadata['proportion_thresholds'].append('0.99 <= prop < 1.17')
                return (
                    'luna_jupiter_balanced',
                    'balanced_emocional_racional',
                    metadata
                )
        
        # ====================================================================
        # NewFeature.md líneas 49-50: sol_neptuno_combined o luna_jupiter_combined >= 1.17
        # ====================================================================
        # "sol_neptuno_combined o luna_jupiter_combined y proporción facial igual o mayor a 1.17:
        #  Aunque suelen ser sociables y divertidos..."
        
        elif diagnosis in ['sol_neptuno_combined', 'luna_jupiter_combined']:
            if proportion >= 1.17:
                # NewFeature.md línea 49: proporción igual o mayor a 1.17
                metadata['proportion_thresholds'].append('>= 1.17')
                return (
                    'espiritual_creativo',
                    'espiritual_idealista_abstracto',
                    metadata
                )
        
        # ====================================================================
        # NewFeature.md líneas 51-53: mercurio_triangular
        # ====================================================================
        # "mercurio_triangular y proporción facial menor a 1.17: rápidos de mente..."
        # "mercurio_triangular y poroporción facial igual o mayor a 1.17: filosóficos..."
        
        elif diagnosis == 'mercurio_triangular':
            if proportion < 1.17:
                # NewFeature.md línea 51: proporción menor a 1.17
                metadata['proportion_thresholds'].append('< 1.17')
                return (
                    'mercurio_rapido',
                    'rapido_volatil_comunicativo',
                    metadata
                )
            else:
                # NewFeature.md línea 53: proporción igual o mayor a 1.17
                metadata['proportion_thresholds'].append('>= 1.17')
                return (
                    'mercurio_filosofico',
                    'filosofico_ordenado_comunicativo',
                    metadata
                )
        
        # Si no hay regla específica, retornar diagnóstico original
        return (diagnosis, 'default_narrative', metadata)
    
    # ========================================================================
    # NewFeature.md: Proporción Frente (líneas 55-69)
    # ========================================================================
    
    def analyze_frente_proportion(
        self,
        diagnosis: str,
        proportion: float,
        confidence: float
    ) -> Tuple[str, str, Dict]:
        """
        NewFeature.md líneas 55-69: Splitting de diagnósticos por Proporción Frente.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 55-69
        
        Umbral principal: 0.35 (líneas 57, 59)
        
        Args:
            diagnosis: Diagnóstico base de FRENTE
            proportion: Proporción Frente (Derecha e Izquierda)
            confidence: Nivel de confianza del diagnóstico
            
        Returns:
            Tuple[adjusted_diagnosis, narrative_key, metadata]
            
        Example:
            # NewFeature.md línea 57: solar_lunar_combined con proporción > 0.35
            diag, narrative, meta = analyzer.analyze_frente_proportion(
                'solar_lunar_combined', 0.40, 0.75
            )
            # diag = 'solar_lunar_emotivo'
        """
        metadata = {
            'original_diagnosis': diagnosis,
            'proportion': proportion,
            'confidence': confidence,
            'proportion_thresholds': []
        }
        
        # ====================================================================
        # NewFeature.md líneas 57-59: solar_lunar_combined
        # ====================================================================
        # Línea 55: "solo tomar solar_lunar_combined si está presente diagnóstico
        #            de tercio proporción frente derecha e izquierda"
        # Línea 57: "solar_lunar_combined y proporción frente mayor a .35:
        #            Persona emotiva, exigentes..."
        # Línea 59: "solar_lunar_combined y proporción Frente menor a .35:
        #            Persona con emociones en frecuente cambio..."
        
        if diagnosis == 'solar_lunar_combined':
            if proportion > 0.35:
                # NewFeature.md línea 57: proporción mayor a 0.35
                metadata['proportion_thresholds'].append('> 0.35')
                return (
                    'solar_lunar_emotivo',
                    'emotivo_liderazgo_analitico',
                    metadata
                )
            else:
                # NewFeature.md línea 59: proporción menor a 0.35
                metadata['proportion_thresholds'].append('<= 0.35')
                return (
                    'solar_lunar_creativo',
                    'emociones_cambiantes_creativo',
                    metadata
                )
        
        # ====================================================================
        # NewFeature.md líneas 61-69: Otros diagnósticos de FRENTE
        # ====================================================================
        # Estos no tienen splitting por proporción, solo umbral general 18%
        
        elif diagnosis == 'mercurio_triangulo':
            # NewFeature.md línea 61: "mercurio_triangulo: persona creativa"
            return (diagnosis, 'persona_creativa', metadata)
        
        elif diagnosis == 'marte_rectangular':
            # NewFeature.md línea 63: "martes rectangular: Persona orientada a accionar"
            return (diagnosis, 'orientado_accion_resultados', metadata)
        
        elif diagnosis == 'jupiter_amplio_base_ancha':
            # NewFeature.md línea 65: "jupiter_amplio_base_ancha: persona capaz de ver panorama amplio"
            return (diagnosis, 'panorama_amplio_enjuicia', metadata)
        
        elif diagnosis == 'neptuno_combined':
            # NewFeature.md línea 67: "neptuno_combined: Persona con gran capacidad de análisis"
            return (diagnosis, 'analisis_intuitivo_abstracto', metadata)
        
        elif diagnosis in ['venus_corazon', 'trapezoide_angosto']:
            # NewFeature.md línea 69: "Venus Corazon O Trapezoide Angosto: persona interesada en bienestar"
            return (diagnosis, 'bienestar_otros_carismatico', metadata)
        
        # Si no hay regla específica, retornar diagnóstico original
        return (diagnosis, 'default_narrative', metadata)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def get_proportion_category(self, proportion: float) -> str:
        """
        NewFeature.md: Categoriza una proporción según los umbrales definidos.
        
        Args:
            proportion: Valor de proporción
            
        Returns:
            Categoría ('muy_bajo', 'bajo', 'medio', 'alto')
        """
        if proportion < 0.99:
            return 'bajo'
        elif proportion < 1.17:
            return 'medio'
        else:
            return 'alto'

