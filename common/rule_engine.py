# ============================================================================
# NewFeature.md: Rule Engine - Reglas de omisión y coincidencia (líneas 81-503)
# ============================================================================
# Motor de reglas para manejar:
# - Reglas de omisión (cejas, tercios, etc.)
# - Reglas de coincidencia (nariz, protrusión ocular, oreja)
# - Reglas de lateralización (izquierda/derecha)
#
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
# ============================================================================

from typing import Dict, Optional, List


class RuleEngine:
    """
    NewFeature.md: Motor de reglas para omisiones, coincidencias y lateralización.
    
    Implementa las reglas especificadas en NewFeature.md que no son umbrales
    de confianza, sino lógica de negocio específica (omitir categorías,
    requerir coincidencia entre lados, etc.)
    """
    
    def __init__(self):
        """Initialize rule engine"""
        pass
    
    # ========================================================================
    # NewFeature.md: Reglas de Omisión (líneas 81-83, 377)
    # ========================================================================
    
    def should_omit_cejas_antropometrico(self) -> bool:
        """
        NewFeature.md línea 81: Cejas antropométrico SIEMPRE se omiten.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 81
        
        Requisito:
        "analisis de cejas (ceja izq y ceja derecha) siempre se omite"
        
        Returns:
            True (siempre)
        """
        # NewFeature.md línea 81: SIEMPRE omitir
        return True
    
    def should_omit_tercios_rostro(self, validation_results: Dict) -> bool:
        """
        NewFeature.md línea 83: Tercios rostro se omiten si validación detecta obstáculos.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 83
        
        Requisito:
        "tercios de rostro se omite si diagnostico en modulo de validacion
         da objeto tapando frente o cabello tapando"
        
        Args:
            validation_results: Resultados del módulo de validación
            
        Returns:
            True si se debe omitir tercios rostro, False otherwise
            
        Example:
            validation = {'detections': ['cabello_tapando_central']}
            should_omit = engine.should_omit_tercios_rostro(validation)
            # should_omit = True
        """
        # NewFeature.md línea 83: Lista de obstáculos que causan omisión
        obstacles = [
            'objeto_tapando_frente',
            'cabello_tapando',
            'cabello_tapando_central',
            'cabello_tapando_i',
            'cabello_tapando_derecho',
        ]
        
        detections = validation_results.get('detections', [])
        
        for obstacle in obstacles:
            if obstacle in detections:
                return True  # NewFeature.md línea 83: Omitir tercios
        
        return False
    
    def should_omit_distancia_trago_antitrago(self) -> bool:
        """
        NewFeature.md línea 377: Distancia trago-antitrago se omite.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 377
        
        Requisito:
        "Distancia trago antitrago se omite"
        
        Returns:
            True (siempre)
        """
        # NewFeature.md línea 377: SIEMPRE omitir
        return True
    
    def should_omit(self, module: str, category: str, context: Optional[Dict] = None) -> bool:
        """
        NewFeature.md: Determina si una categoría debe omitirse.
        
        Método genérico que delega a los métodos específicos según módulo y categoría.
        
        Args:
            module: Nombre del módulo (e.g., 'frontal_antropometrico')
            category: Categoría a evaluar (e.g., 'cejas', 'tercios_rostro')
            context: Contexto adicional (e.g., validation_results)
            
        Returns:
            True si se debe omitir, False otherwise
        """
        if module == 'frontal_antropometrico':
            if category == 'cejas':
                return self.should_omit_cejas_antropometrico()
            elif category == 'tercios_rostro':
                if context and 'validation_results' in context:
                    return self.should_omit_tercios_rostro(context['validation_results'])
        
        elif module == 'profile_antropometrico':
            if category == 'distancia_trago_antitrago':
                return self.should_omit_distancia_trago_antitrago()
        
        return False
    
    # ========================================================================
    # NewFeature.md: Reglas de Coincidencia (líneas 379, 479, 491)
    # ========================================================================
    
    def check_nariz_coincidence(
        self,
        left_profile_nariz: Optional[str],
        right_profile_nariz: Optional[str]
    ) -> Optional[str]:
        """
        NewFeature.md líneas 379-389: Largo de nariz requiere coincidencia entre perfiles.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 379
        
        Requisito:
        "Para analisis de la largo de nariz, si ambos perfiles están presentes,
         ambos tienen que tener el mismo diagnóstico para que sea el diagnóstico final,
         de no ser así, el diagnostico es nariz normal y no se menciona."
        
        Args:
            left_profile_nariz: Diagnóstico de perfil izquierdo
            right_profile_nariz: Diagnóstico de perfil derecho
            
        Returns:
            Diagnóstico final (coincidencia o 'nariz_normal')
            
        Example:
            result = engine.check_nariz_coincidence('nariz_larga', 'nariz_larga')
            # result = 'nariz_larga' (coinciden)
            
            result = engine.check_nariz_coincidence('nariz_larga', 'nariz_corta')
            # result = 'nariz_normal' (no coinciden, línea 389)
        """
        if left_profile_nariz and right_profile_nariz:
            # NewFeature.md línea 379: Ambos presentes -> deben coincidir
            if left_profile_nariz == right_profile_nariz:
                return left_profile_nariz
            else:
                # NewFeature.md línea 389: No coinciden -> "nariz normal"
                return 'nariz_normal'
        elif left_profile_nariz:
            # NewFeature.md línea 380: Solo izquierdo
            return left_profile_nariz
        elif right_profile_nariz:
            # NewFeature.md línea 380: Solo derecho
            return right_profile_nariz
        
        return None
    
    def check_protrusion_ocular_coincidence(
        self,
        left_protrusion: Optional[str],
        right_protrusion: Optional[str]
    ) -> Optional[str]:
        """
        NewFeature.md línea 479: Protrusión ocular requiere coincidencia.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 479
        
        Requisito:
        "Para que protusion ocular se diagnostique, ambos diagnosticos
         tienen que estar presentes y coincidir"
        
        Args:
            left_protrusion: Protrusión ocular izquierda
            right_protrusion: Protrusión ocular derecha
            
        Returns:
            Diagnóstico si ambos presentes Y coinciden, None otherwise
            
        Example:
            result = engine.check_protrusion_ocular_coincidence('positiva', 'positiva')
            # result = 'positiva' (ambos presentes y coinciden)
            
            result = engine.check_protrusion_ocular_coincidence('positiva', 'negativa')
            # result = None (no coinciden)
            
            result = engine.check_protrusion_ocular_coincidence('positiva', None)
            # result = None (no ambos presentes)
        """
        if left_protrusion and right_protrusion:
            # NewFeature.md línea 479: Ambos presentes Y deben coincidir
            if left_protrusion == right_protrusion:
                return left_protrusion
        
        # No coinciden o no ambos presentes -> None
        return None
    
    def check_oreja_largo_coincidence(
        self,
        left_oreja: Optional[str],
        right_oreja: Optional[str]
    ) -> Optional[str]:
        """
        NewFeature.md línea 491: Largo de oreja requiere coincidencia.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 491
        
        Requisito:
        "Para largo de oreja, ambos diagnósticos tienen que estar presentes y coincidir"
        
        Args:
            left_oreja: Largo de oreja izquierda
            right_oreja: Largo de oreja derecha
            
        Returns:
            Diagnóstico si ambos presentes Y coinciden, None otherwise
        """
        if left_oreja and right_oreja:
            # NewFeature.md línea 491: Ambos presentes Y deben coincidir
            if left_oreja == right_oreja:
                return left_oreja
        
        # No coinciden o no ambos presentes -> None
        return None
    
    def requires_coincidence(self, module: str, category: str) -> bool:
        """
        NewFeature.md: Determina si una categoría requiere coincidencia entre lados.
        
        Args:
            module: Nombre del módulo
            category: Categoría a evaluar
            
        Returns:
            True si requiere coincidencia, False otherwise
        """
        # NewFeature.md: Categorías que requieren coincidencia
        coincidence_categories = {
            'profile_antropometrico': ['nariz_largo'],  # línea 379
            'frontal_antropometrico': ['protrusion_ocular', 'oreja_largo'],  # líneas 479, 491
        }
        
        return category in coincidence_categories.get(module, [])
    
    # ========================================================================
    # NewFeature.md: Lateralización (línea 5)
    # ========================================================================
    
    def ensure_lateralization(self, diagnosis: Dict) -> Dict:
        """
        NewFeature.md línea 5: Asegurar indicación de lado derecho o izquierdo.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 5
        
        Requisito:
        "Es vital indicar de dónde viene no solo su categoría si no si es de
         lado derecho o izquierdo ejem; pomulo derecho o pomulo izquierdo (Pml_d, pml_i)"
        
        Args:
            diagnosis: Diagnóstico a validar
            
        Returns:
            Diagnóstico con lateralización asegurada
        """
        # Verificar que el diagnóstico incluya indicador de lado
        if 'side' not in diagnosis or diagnosis['side'] is None:
            raise ValueError(
                "NewFeature.md línea 5: Diagnóstico debe indicar lado (derecho/izquierdo)"
            )
        
        return diagnosis

