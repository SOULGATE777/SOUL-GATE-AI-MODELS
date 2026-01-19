# ============================================================================
# NewFeature.md: Diagnosis Resolver - Combinaciones complejas (líneas 323-428)
# ============================================================================
# Resuelve diagnósticos complejos basados en combinaciones de características:
# - Frente 2 criterios (antropométrico vs morfológico)
# - Mandíbula + Mentón combinaciones
# - Nariz combinada (corta + punta arriba)
#
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
# ============================================================================

from typing import Dict, Optional, Tuple


class DiagnosisResolver:
    """
    NewFeature.md: Resuelve diagnósticos complejos basados en combinaciones.
    
    Implementa la lógica de resolución para casos donde un diagnóstico final
    depende de múltiples características combinadas o de criterios específicos
    de priorización entre módulos.
    """
    
    def __init__(self):
        """Initialize diagnosis resolver"""
        pass
    
    # ========================================================================
    # NewFeature.md: Frente Perfil - 2 Criterios (líneas 323-373)
    # ========================================================================
    
    def resolve_frente_perfil(
        self, 
        left_profile: Optional[Dict],
        right_profile: Optional[Dict],
        antropometrico_result: Optional[str],
        morfologico_result: Optional[str]
    ) -> Optional[str]:
        """
        NewFeature.md líneas 323-373: Frente perfil con 2 criterios (antropométrico y morfológico).
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 323-330
        
        Requisitos:
        - Escenario 1 (línea 325): Solo perfil Izquierdo -> criterio antropométrico
        - Escenario 2 (línea 327): Solo perfil derecho -> criterio morfológico
        - Escenario 3 (línea 329): Ambos perfiles -> coincidir o priorizar derecho
        
        Args:
            left_profile: Datos del perfil izquierdo (si existe)
            right_profile: Datos del perfil derecho (si existe)
            antropometrico_result: Resultado antropométrico
            morfologico_result: Resultado morfológico
            
        Returns:
            Diagnóstico final de frente o None
            
        Example:
            # NewFeature.md línea 325: Escenario 1 - Solo izquierdo
            result = resolver.resolve_frente_perfil(
                left_profile={'exists': True},
                right_profile=None,
                antropometrico_result='frente_vertical',
                morfologico_result=None
            )
            # result = 'frente_vertical' (usa antropométrico)
        """
        # ====================================================================
        # NewFeature.md línea 325: Escenario 1 - Solo perfil Izquierdo
        # ====================================================================
        # "Escenario 1: Solo perfil Izquierdo - se toma criterio antropométrico"
        
        if left_profile and not right_profile:
            return antropometrico_result
        
        # ====================================================================
        # NewFeature.md línea 327: Escenario 2 - Solo perfil derecho
        # ====================================================================
        # "Escenario 2: Solo perfil derecho - se toma criterio morfológico"
        
        elif right_profile and not left_profile:
            return morfologico_result
        
        # ====================================================================
        # NewFeature.md línea 329: Escenario 3 - Ambos perfiles
        # ====================================================================
        # "Escenario 3: Ambos perfiles - si ambos coinciden en diagnósticos
        #  se procede con dicho diagnostico, si no se coincide se difiere
        #  a diagnostico de perfil derecho"
        
        elif left_profile and right_profile:
            if antropometrico_result == morfologico_result:
                # Coinciden -> usar cualquiera (son iguales)
                return antropometrico_result
            else:
                # No coinciden -> priorizar perfil derecho (morfológico)
                return morfologico_result
        
        return None
    
    # ========================================================================
    # NewFeature.md: Mandíbula + Mentón Combinaciones (líneas 405-428)
    # ========================================================================
    
    def resolve_mandibula_menton(
        self,
        mandibula: str,
        menton: str,
        frente: Optional[str] = None
    ) -> Dict:
        """
        NewFeature.md líneas 405-428: Combinaciones mandíbula + mentón.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 405-428
        
        NOTA: Colorimetría (piel) NO implementada en esta iteración.
              Las reglas que dependen de color de piel se implementarán
              en iteración posterior con Temperamento.
        
        Args:
            mandibula: Tipo de mandíbula ('biliosa', 'linfatica', 'larga_vertical', 'nerviosa')
            menton: Tipo de mentón ('sanguineo', 'bilioso_linfatico', 'nervioso')
            frente: Tipo de frente (opcional)
            
        Returns:
            Dict con diagnóstico combinado y narrativa
            
        Example:
            # NewFeature.md líneas 409-410: mandibula sanguineo
            result = resolver.resolve_mandibula_menton(
                mandibula='biliosa',
                menton='sanguineo',
                frente='frente_vertical'
            )
            # result = {
            #   'temperament': 'bilioso_impulsivo',
            #   'narrative_key': 'bilioso_con_impulso_sanguineo'
            # }
        """
        result = {
            'mandibula': mandibula,
            'menton': menton,
            'frente': frente,
            'temperament': None,
            'narrative_key': None,
            'requires_colorimetry': False  # Flag para indicar que falta colorimetría
        }
        
        # ====================================================================
        # NewFeature.md líneas 409-410: menton sanguineo
        # ====================================================================
        # "menton sanguineo - person con impulso a acturar rapidamente"
        
        if menton == 'sanguineo':
            result['narrative_key'] = 'impulso_actuar_rapidamente'
        
        # ====================================================================
        # NewFeature.md líneas 411-412: menton biloso/linfatico
        # ====================================================================
        # "menton biloso/linfatico - no se menciona"
        
        elif menton == 'biloso_linfatico':
            result['narrative_key'] = None  # No se menciona
        
        # ====================================================================
        # NewFeature.md líneas 413-414: menton nervioso
        # ====================================================================
        # "menton nervioso - persona que piensa mucho antes de actuar"
        
        elif menton == 'nervioso':
            result['narrative_key'] = 'piensa_antes_actuar'
        
        # ====================================================================
        # NewFeature.md líneas 417-428: Análisis de Mandíbula
        # ====================================================================
        
        if mandibula == 'biliosa':
            # NewFeature.md línea 423: "mandibula Bilosa - persona con mucha fuerza
            #                           para continuar con lo que inicia"
            result['temperament'] = 'bilioso'
            
        elif mandibula == 'sanguinea':
            # NewFeature.md línea 425: "mandíbula Sanguínea - persona con mucha fuerza
            #                           en un inicio o primer impulso pero que le batalla
            #                           para continuar"
            result['temperament'] = 'sanguineo'
            
        elif mandibula == 'intermedia':
            # NewFeature.md línea 427: "mandibula intermedia sanguineo/bilosa - no se menciona"
            result['temperament'] = 'intermedio'
            result['narrative_key'] = None
        
        # NOTA: Las combinaciones complejas de líneas 429-477 que incluyen color de piel
        # se implementarán en iteración posterior (FASE TEMPERAMENTO)
        
        return result
    
    # ========================================================================
    # NewFeature.md: Nariz Combinada (líneas 396-401)
    # ========================================================================
    
    def resolve_nariz_combined(
        self,
        nariz_largo: Optional[str],
        nariz_angulo: Optional[str]
    ) -> Optional[str]:
        """
        NewFeature.md líneas 396-401: Nariz corta + punta arriba = impaciente.
        
        Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 401
        
        Requisito:
        "nariz corta + punta de nariz hacia arriba: persona impaciente"
        
        Args:
            nariz_largo: Análisis de largo de nariz ('nariz_corta', 'nariz_normal', 'nariz_larga')
            nariz_angulo: Análisis de ángulo ('punta_hacia_arriba', 'punta_promedio', 'punta_hacia_abajo')
            
        Returns:
            Diagnóstico combinado o None
            
        Example:
            # NewFeature.md línea 401
            result = resolver.resolve_nariz_combined('nariz_corta', 'punta_hacia_arriba')
            # result = 'persona_impaciente'
            
            result = resolver.resolve_nariz_combined('nariz_larga', 'punta_hacia_arriba')
            # result = None (no cumple la combinación específica)
        """
        # ====================================================================
        # NewFeature.md línea 385: nariz corta sin punta arriba
        # ====================================================================
        # "nariz corta: impaciente; solo si diagnóstico de nariz punta hacia arriba
        #  tambien esta presente"
        
        if nariz_largo == 'nariz_corta':
            if nariz_angulo == 'punta_hacia_arriba':
                # NewFeature.md línea 401: Combinación específica
                return 'persona_impaciente'
            else:
                # Nariz corta sin punta hacia arriba -> no se menciona como impaciente
                return None
        
        # ====================================================================
        # NewFeature.md línea 387: nariz protruyente
        # ====================================================================
        # "nariz protruyente: persona inteligente"
        
        elif nariz_largo == 'nariz_protruyente':
            return 'persona_inteligente'
        
        # ====================================================================
        # NewFeature.md línea 389: nariz normal
        # ====================================================================
        # "nariz normal: no se menciona"
        
        elif nariz_largo == 'nariz_normal':
            return None  # No se menciona
        
        return None
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def combine_diagnoses(self, diagnoses: list) -> Dict:
        """
        NewFeature.md: Combina múltiples diagnósticos en un resultado consolidado.
        
        Args:
            diagnoses: Lista de diagnósticos individuales
            
        Returns:
            Dict con diagnósticos combinados y metadata
        """
        return {
            'combined': diagnoses,
            'count': len(diagnoses),
            'primary': diagnoses[0] if diagnoses else None
        }

    
    # ========================================================================
    # NewFeature.md: Temperamento Completo (líneas 429-477)
    # ========================================================================
    
    def resolve_temperament_complete(
        self,
        profile_results: Dict,
        eye_colorimetry: Dict,
        palm_colorimetry: Dict,
        temperament_service_url: str = 'http://localhost:4060'
    ) -> Dict:
        """
        NUEVO MÉTODO: Consolidación completa de temperamento.
        
        NewFeature.md: Combina todos los análisis para temperamento.
        
        Combina:
        1. Mandíbula + Mentón del perfil morfológico
        2. Frente (2 criterios) del perfil
        3. Tono de piel (facial + palmas)
        4. Colorimetría de ojos
        
        Args:
            profile_results: Resultados del análisis de perfil (contiene mandíbula, mentón, frente)
            eye_colorimetry: Resultados del análisis de colorimetría de ojos
            palm_colorimetry: Resultados del análisis de colorimetría de palmas
            temperament_service_url: URL del servicio de temperamento (default: localhost:4060)
        
        Returns:
            Dict con análisis completo de temperamento, colorimetría de ojos y palmas, y tono de piel
        """
        import os
        try:
            import requests
        except ImportError:
            return {
                'error': 'requests library not available',
                'message': 'Install requests to use temperament resolution'
            }
        
        from common.colorimetry_analyzer import ColorimetryAnalyzer
        
        colorimetry = ColorimetryAnalyzer()
        
        try:
            # 1. Extraer mandíbula/mentón/frente de profile_results
            mandibula = profile_results.get('mandibula', 'unknown')
            menton = profile_results.get('menton', 'unknown')
            
            # 2. Resolver frente con 2 criterios (método existente)
            frente = self.resolve_frente_perfil(
                left_profile=profile_results.get('left_profile'),
                right_profile=profile_results.get('right_profile'),
                antropometrico_result=profile_results.get('frente_antropometrico'),
                morfologico_result=profile_results.get('frente_morfologico')
            )
            
            if not frente:
                frente = 'unknown'
            
            # 3. Analizar tono de piel combinado
            face_rgb = profile_results.get('face_avg_rgb')
            palm_categories = palm_colorimetry.get('temperament_categories', [])
            
            skin_tone = colorimetry.analyze_skin_tone(
                face_rgb=face_rgb,
                palm_categories=palm_categories
            )
            
            # 4. Llamar a servicio de temperamento
            # Usar variable de entorno si está disponible
            service_url = os.getenv('TEMPERAMENT_SERVICE_URL', temperament_service_url)
            
            response = requests.post(
                f"{service_url}/api/v1/analyze-temperament",
                json={
                    'mandibula': mandibula,
                    'menton': menton,
                    'frente': frente,
                    'skin_tone': skin_tone if skin_tone.get('tone') else None
                },
                timeout=10
            )
            
            if response.status_code == 200:
                temperament = response.json()
            else:
                temperament = {
                    'error': f'Temperament service returned status {response.status_code}',
                    'message': response.text
                }
            
            return {
                'temperament': temperament,
                'eye_colorimetry': eye_colorimetry,
                'palm_colorimetry': palm_colorimetry,
                'skin_tone': skin_tone,
                'service_url': service_url
            }
        
        except requests.RequestException as e:
            return {
                'error': 'Failed to connect to temperament service',
                'message': str(e),
                'service_url': temperament_service_url,
                'suggestion': 'Ensure temperament service is running at ' + temperament_service_url
            }
        except Exception as e:
            return {
                'error': 'Unexpected error in temperament resolution',
                'message': str(e)
            }
