# ============================================================================
# NewFeature.md: Threshold Configuration - Single Source of Truth
# ============================================================================
# Configuración centralizada de TODOS los umbrales para TODOS los módulos ML.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md
#
# Este archivo contiene la configuración completa de umbrales especificados
# en NewFeature.md, organizados por módulo y con referencias explícitas
# a las líneas del documento fuente.
# ============================================================================

from .threshold_validator import ModuleType, ThresholdRule


# ============================================================================
# NewFeature.md: Modulo Espejo (personalidad) - FRENTE (líneas 13-23, 55-69)
# ============================================================================
# Umbrales de confianza para diagnósticos de FRENTE.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md línea 19
#
# Requisito:
# "Si predicción de frente es mayor al 18% se toma como diagnostico certero,
#  de lo contrario, se omite."
# ============================================================================

ESPEJO_FRENTE_RULES = {
    'solar_lunar_combined': ThresholdRule(
        diagnosis_name='solar_lunar_combined',
        threshold=0.18,  # NewFeature.md línea 19: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 19: 18% minimum threshold for FRENTE'
    ),
    'neptuno_combined': ThresholdRule(
        diagnosis_name='neptuno_combined',
        threshold=0.18,  # NewFeature.md línea 19: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 19: 18% minimum threshold for FRENTE'
    ),
    'jupiter_amplio_base_ancha': ThresholdRule(
        diagnosis_name='jupiter_amplio_base_ancha',
        threshold=0.18,  # NewFeature.md línea 19: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 19: 18% minimum threshold for FRENTE'
    ),
    'marte_rectangular': ThresholdRule(
        diagnosis_name='marte_rectangular',
        threshold=0.18,  # NewFeature.md línea 19: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 19: 18% minimum threshold for FRENTE'
    ),
    'venus_corazon_o_trapezoide_angosto': ThresholdRule(
        diagnosis_name='venus_corazon_o_trapezoide_angosto',
        threshold=0.18,  # NewFeature.md línea 19: "mayor al 18%" (no exception for FRENTE)
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 19: 18% minimum (no exception for FRENTE, only ROSTRO)'
    ),
    'mercurio_triangulo': ThresholdRule(
        diagnosis_name='mercurio_triangulo',
        threshold=0.18,  # NewFeature.md línea 19: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 19: 18% minimum threshold for FRENTE'
    ),
}


# ============================================================================
# NewFeature.md: Modulo Espejo (personalidad) - ROSTRO (líneas 13-23, 29-54)
# ============================================================================
# Umbrales de confianza para diagnósticos de ROSTRO (rostro_menton region).
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 17, 22-23
#
# Requisito línea 17:
# "Si predicción de rostro es mayor al 18% se toma como diagnostico certero,
#  de lo contrario, se omite."
#
# Excepciones (líneas 22-23):
# "Pluton Hexagonal (cuyo umbral es mayor a 7%)"
# "Venus Corazon (mayor a 40%)"
# ============================================================================

ESPEJO_ROSTRO_RULES = {
    # ========================================================================
    # EXCEPCIONES: Venus Corazón y Plutón Hexagonal
    # ========================================================================
    
    'venus_corazon': ThresholdRule(
        diagnosis_name='venus_corazon',
        threshold=0.40,  # NewFeature.md línea 23: "Venus Corazon (mayor a 40%)"
        rule_type='minimum',
        is_exception=True,
        description='NewFeature.md línea 23: Venus Corazon exception - requires >40% confidence'
    ),
    
    'pluton_hexagonal': ThresholdRule(
        diagnosis_name='pluton_hexagonal',
        threshold=0.07,  # NewFeature.md línea 22: "Pluton Hexagonal (cuyo umbral es mayor a 7%)"
        rule_type='minimum',
        is_exception=True,
        description='NewFeature.md línea 22: Pluton Hexagonal exception - requires >7% confidence'
    ),
    
    # ========================================================================
    # Resto de diagnósticos: Umbral general 18%
    # ========================================================================
    
    'saturno_trapezoide_base_angosta': ThresholdRule(
        diagnosis_name='saturno_trapezoide_base_angosta',
        threshold=0.18,  # NewFeature.md línea 17: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 17: 18% minimum threshold for ROSTRO'
    ),
    
    'luna_jupiter_combined': ThresholdRule(
        diagnosis_name='luna_jupiter_combined',
        threshold=0.18,  # NewFeature.md línea 17: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 17: 18% minimum threshold for ROSTRO'
    ),
    
    'mercurio_triangular': ThresholdRule(
        diagnosis_name='mercurio_triangular',
        threshold=0.18,  # NewFeature.md línea 17: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 17: 18% minimum threshold for ROSTRO'
    ),
    
    'marte_tierra_rectangulo': ThresholdRule(
        diagnosis_name='marte_tierra_rectangulo',
        threshold=0.18,  # NewFeature.md línea 17: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 17: 18% minimum threshold for ROSTRO'
    ),
    
    'sol_neptuno_combined': ThresholdRule(
        diagnosis_name='sol_neptuno_combined',
        threshold=0.18,  # NewFeature.md línea 17: "mayor al 18%"
        rule_type='minimum',
        is_exception=False,
        description='NewFeature.md línea 17: 18% minimum threshold for ROSTRO'
    ),
}


# ============================================================================
# NewFeature.md: Modulo frontal morfologico (líneas 119-270)
# ============================================================================
# Umbrales de confianza para diagnósticos morfológicos frontales.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 119-270
#
# Regla general (línea 121):
# "Todo umbral en porcentaje entre paréntesis significa que el diagnóstico certero
#  tiene que ser mayor a ese porcentaje, de no ser así se omite."
#
# Regla adicional (línea 73):
# "si no existe umbral, se toma el de mayor porcentaje tenga mínimo 15 puntos
#  de porcentaje mayor a los demás"
# ============================================================================

FRONTAL_MORFOLOGICO_RULES = {
    # ========================================================================
    # NewFeature.md líneas 123-131: Cejas (cj_d, cj_i)
    # ========================================================================
    'cj_d_cv': ThresholdRule(
        diagnosis_name='cj_d_cv',
        threshold=0.50,  # NewFeature.md línea 127: "cV (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 127: Ceja derecha curva 50%'
    ),
    'cj_d_el': ThresholdRule(
        diagnosis_name='cj_d_el',
        threshold=0.50,  # NewFeature.md línea 129: "el (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 129: Ceja derecha inclinada 50%'
    ),
    'cj_d_rc': ThresholdRule(
        diagnosis_name='cj_d_rc',
        threshold=0.50,  # NewFeature.md línea 131: "rc (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 131: Ceja derecha recta 50%'
    ),
    'cj_i_cv': ThresholdRule(
        diagnosis_name='cj_i_cv',
        threshold=0.50,  # NewFeature.md línea 127: "cV (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 127: Ceja izquierda curva 50%'
    ),
    'cj_i_el': ThresholdRule(
        diagnosis_name='cj_i_el',
        threshold=0.50,  # NewFeature.md línea 129: "el (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 129: Ceja izquierda inclinada 50%'
    ),
    'cj_i_rc': ThresholdRule(
        diagnosis_name='cj_i_rc',
        threshold=0.50,  # NewFeature.md línea 131: "rc (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 131: Ceja izquierda recta 50%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 133-141: Entrecejo
    # ========================================================================
    'entrecejo_uniceja': ThresholdRule(
        diagnosis_name='entrecejo_uniceja',
        threshold=0.60,  # NewFeature.md línea 139: "uniceja (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 139: Uniceja 60%'
    ),
    'entrecejo_lineas_verticales': ThresholdRule(
        diagnosis_name='entrecejo_lineas_verticales',
        threshold=0.60,  # NewFeature.md línea 141: "lineas_verticales (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 141: Líneas verticales 60%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 143-149: Párpado (parpado_i, parpado_dr)
    # ========================================================================
    'parpado_i_ptosis': ThresholdRule(
        diagnosis_name='parpado_i_ptosis',
        threshold=0.60,  # NewFeature.md línea 147: "ptosis (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 147: Ptosis palpebral izquierdo 60%'
    ),
    'parpado_i_pliegue': ThresholdRule(
        diagnosis_name='parpado_i_pliegue',
        threshold=0.60,  # NewFeature.md línea 149: "pliegue (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 149: Pliegue palpebral izquierdo 60%'
    ),
    'parpado_dr_ptosis': ThresholdRule(
        diagnosis_name='parpado_dr_ptosis',
        threshold=0.60,  # NewFeature.md línea 147: "ptosis (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 147: Ptosis palpebral derecho 60%'
    ),
    'parpado_dr_pliegue': ThresholdRule(
        diagnosis_name='parpado_dr_pliegue',
        threshold=0.60,  # NewFeature.md línea 149: "pliegue (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 149: Pliegue palpebral derecho 60%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 151-163: Ojo (oj_d, oj_i)
    # ========================================================================
    'oj_d_al': ThresholdRule(
        diagnosis_name='oj_d_al',
        threshold=0.22,  # NewFeature.md línea 157: "al (22%)"
        rule_type='minimum',
        description='NewFeature.md línea 157: Ojo derecho almendrado 22%'
    ),
    'oj_d_fr': ThresholdRule(
        diagnosis_name='oj_d_fr',
        threshold=0.35,  # NewFeature.md línea 159: "fr (35%)"
        rule_type='minimum',
        description='NewFeature.md línea 159: Ojo derecho fruncido 35%'
    ),
    'oj_d_md': ThresholdRule(
        diagnosis_name='oj_d_md',
        threshold=0.70,  # NewFeature.md línea 161: "md (70%)"
        rule_type='minimum',
        description='NewFeature.md línea 161: Ojo derecho media luna arriba 70%'
    ),
    'oj_d_md_a': ThresholdRule(
        diagnosis_name='oj_d_md_a',
        threshold=0.22,  # NewFeature.md línea 163: "md_a (22%)"
        rule_type='minimum',
        description='NewFeature.md línea 163: Ojo derecho media luna abajo 22%'
    ),
    'oj_i_al': ThresholdRule(
        diagnosis_name='oj_i_al',
        threshold=0.22,  # NewFeature.md línea 157: "al (22%)"
        rule_type='minimum',
        description='NewFeature.md línea 157: Ojo izquierdo almendrado 22%'
    ),
    'oj_i_fr': ThresholdRule(
        diagnosis_name='oj_i_fr',
        threshold=0.35,  # NewFeature.md línea 159: "fr (35%)"
        rule_type='minimum',
        description='NewFeature.md línea 159: Ojo izquierdo fruncido 35%'
    ),
    'oj_i_md': ThresholdRule(
        diagnosis_name='oj_i_md',
        threshold=0.70,  # NewFeature.md línea 161: "md (70%)"
        rule_type='minimum',
        description='NewFeature.md línea 161: Ojo izquierdo media luna arriba 70%'
    ),
    'oj_i_md_a': ThresholdRule(
        diagnosis_name='oj_i_md_a',
        threshold=0.22,  # NewFeature.md línea 163: "md_a (22%)"
        rule_type='minimum',
        description='NewFeature.md línea 163: Ojo izquierdo media luna abajo 22%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 165-175: Oído (o_d, o_i)
    # ========================================================================
    'o_d_sp_sl': ThresholdRule(
        diagnosis_name='o_d_sp_sl',
        threshold=0.30,  # NewFeature.md línea 169: "sp_sl (30%)"
        rule_type='minimum',
        description='NewFeature.md línea 169: Oído derecho tercio superior salido 30%'
    ),
    'o_d_sl': ThresholdRule(
        diagnosis_name='o_d_sl',
        threshold=0.33,  # NewFeature.md línea 171: "sl (33%)"
        rule_type='minimum',
        description='NewFeature.md línea 171: Oído derecho salido 33%'
    ),
    'o_d_pg': ThresholdRule(
        diagnosis_name='o_d_pg',
        threshold=0.25,  # NewFeature.md línea 173: "pg (25%)"
        rule_type='minimum',
        description='NewFeature.md línea 173: Oído derecho pegado 25%'
    ),
    'o_d_pm': ThresholdRule(
        diagnosis_name='o_d_pm',
        threshold=0.25,  # NewFeature.md línea 175: "pm (25%)"
        rule_type='minimum',
        description='NewFeature.md línea 175: Oído derecho promedio 25%'
    ),
    'o_i_sp_sl': ThresholdRule(
        diagnosis_name='o_i_sp_sl',
        threshold=0.30,  # NewFeature.md línea 169: "sp_sl (30%)"
        rule_type='minimum',
        description='NewFeature.md línea 169: Oído izquierdo tercio superior salido 30%'
    ),
    'o_i_sl': ThresholdRule(
        diagnosis_name='o_i_sl',
        threshold=0.33,  # NewFeature.md línea 171: "sl (33%)"
        rule_type='minimum',
        description='NewFeature.md línea 171: Oído izquierdo salido 33%'
    ),
    'o_i_pg': ThresholdRule(
        diagnosis_name='o_i_pg',
        threshold=0.25,  # NewFeature.md línea 173: "pg (25%)"
        rule_type='minimum',
        description='NewFeature.md línea 173: Oído izquierdo pegado 25%'
    ),
    'o_i_pm': ThresholdRule(
        diagnosis_name='o_i_pm',
        threshold=0.25,  # NewFeature.md línea 175: "pm (25%)"
        rule_type='minimum',
        description='NewFeature.md línea 175: Oído izquierdo promedio 25%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 177-185: Nariz grosor
    # ========================================================================
    'nariz_nrml': ThresholdRule(
        diagnosis_name='nariz_nrml',
        threshold=0.65,  # NewFeature.md línea 181: "nrml (65%)"
        rule_type='minimum',
        description='NewFeature.md línea 181: Nariz normal 65%'
    ),
    'nariz_grueso': ThresholdRule(
        diagnosis_name='nariz_grueso',
        threshold=0.22,  # NewFeature.md línea 183: "grueso (22%)"
        rule_type='minimum',
        description='NewFeature.md línea 183: Nariz gruesa 22%'
    ),
    'nariz_delgada': ThresholdRule(
        diagnosis_name='nariz_delgada',
        threshold=0.14,  # NewFeature.md línea 185: "delgada (14%)"
        rule_type='minimum',
        description='NewFeature.md línea 185: Nariz delgada 14%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 187-193: Punta de nariz
    # ========================================================================
    'n_rd': ThresholdRule(
        diagnosis_name='n_rd',
        threshold=0.50,  # NewFeature.md línea 189: "rd (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 189: Punta nariz redondeada 50%'
    ),
    'n_pn': ThresholdRule(
        diagnosis_name='n_pn',
        threshold=0.60,  # NewFeature.md línea 193: "pn (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 193: Punta nariz puntiaguda 60%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 197-205: Pómulo (Pml_d, Pml_i)
    # ========================================================================
    'pml_d_pm': ThresholdRule(
        diagnosis_name='pml_d_pm',
        threshold=0.60,  # NewFeature.md línea 203: "pm (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 203: Pómulo derecho promedio 60%'
    ),
    'pml_d_pl': ThresholdRule(
        diagnosis_name='pml_d_pl',
        threshold=0.60,  # NewFeature.md línea 205: "pl (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 205: Pómulo derecho plano 60%'
    ),
    'pml_i_pm': ThresholdRule(
        diagnosis_name='pml_i_pm',
        threshold=0.60,  # NewFeature.md línea 203: "pm (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 203: Pómulo izquierdo promedio 60%'
    ),
    'pml_i_pl': ThresholdRule(
        diagnosis_name='pml_i_pl',
        threshold=0.60,  # NewFeature.md línea 205: "pl (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 205: Pómulo izquierdo plano 60%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 207-217: Cachete (Cch_d, Cch_i)
    # ========================================================================
    'cch_d_ll': ThresholdRule(
        diagnosis_name='cch_d_ll',
        threshold=0.60,  # NewFeature.md línea 211: "ll (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 211: Cachete derecho lleno 60%'
    ),
    'cch_d_pl': ThresholdRule(
        diagnosis_name='cch_d_pl',
        threshold=0.80,  # NewFeature.md línea 213: "pl (80%)"
        rule_type='minimum',
        description='NewFeature.md línea 213: Cachete derecho plano 80%'
    ),
    'cch_d_planos': ThresholdRule(  # Alias para modelo que genera "planos"
        diagnosis_name='cch_d_planos',
        threshold=0.80,  # NewFeature.md línea 213: "pl (80%)"
        rule_type='minimum',
        description='NewFeature.md línea 213: Cachete derecho plano 80% (alias)'
    ),
    'cch_d_hn': ThresholdRule(
        diagnosis_name='cch_d_hn',
        threshold=0.60,  # NewFeature.md línea 215: "hn (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 215: Cachete derecho hundido 60%'
    ),
    'cch_d_lineas_sonriza': ThresholdRule(
        diagnosis_name='cch_d_lineas_sonriza',
        threshold=0.80,  # NewFeature.md línea 217: "lineas_sonriza (80%)"
        rule_type='minimum',
        description='NewFeature.md línea 217: Cachete derecho líneas sonrisa 80%'
    ),
    'cch_i_ll': ThresholdRule(
        diagnosis_name='cch_i_ll',
        threshold=0.60,  # NewFeature.md línea 211: "ll (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 211: Cachete izquierdo lleno 60%'
    ),
    'cch_i_pl': ThresholdRule(
        diagnosis_name='cch_i_pl',
        threshold=0.80,  # NewFeature.md línea 213: "pl (80%)"
        rule_type='minimum',
        description='NewFeature.md línea 213: Cachete izquierdo plano 80%'
    ),
    'cch_i_planos': ThresholdRule(  # Alias para modelo que genera "planos"
        diagnosis_name='cch_i_planos',
        threshold=0.80,  # NewFeature.md línea 213: "pl (80%)"
        rule_type='minimum',
        description='NewFeature.md línea 213: Cachete izquierdo plano 80% (alias)'
    ),
    'cch_i_hn': ThresholdRule(
        diagnosis_name='cch_i_hn',
        threshold=0.60,  # NewFeature.md línea 215: "hn (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 215: Cachete izquierdo hundido 60%'
    ),
    'cch_i_lineas_sonriza': ThresholdRule(
        diagnosis_name='cch_i_lineas_sonriza',
        threshold=0.80,  # NewFeature.md línea 217: "lineas_sonriza (80%)"
        rule_type='minimum',
        description='NewFeature.md línea 217: Cachete izquierdo líneas sonrisa 80%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 219-229: Forma de boca (bc)
    # ========================================================================
    'bc_lunar': ThresholdRule(
        diagnosis_name='bc_lunar',
        threshold=0.60,  # NewFeature.md línea 223: "lunar (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 223: Boca lunar 60%'
    ),
    'bc_solar': ThresholdRule(
        diagnosis_name='bc_solar',
        threshold=0.30,  # NewFeature.md línea 225: "solar (30%)"
        rule_type='minimum',
        description='NewFeature.md línea 225: Boca solar 30%'
    ),
    'bc_mercurial': ThresholdRule(
        diagnosis_name='bc_mercurial',
        threshold=0.50,  # NewFeature.md línea 227: "mercurial (50%)"
        rule_type='minimum',
        description='NewFeature.md línea 227: Boca mercurial 50%'
    ),
    'bc_pursed': ThresholdRule(
        diagnosis_name='bc_pursed',
        threshold=0.30,  # NewFeature.md línea 229: "pursed (30%)"
        rule_type='minimum',
        description='NewFeature.md línea 229: Boca pursed 30%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 231-239: Arco de cupido (ac_d, ac_i)
    # ========================================================================
    'ac_d_nd': ThresholdRule(
        diagnosis_name='ac_d_nd',
        threshold=0.70,  # NewFeature.md línea 235: "nd (70%)"
        rule_type='minimum',
        description='NewFeature.md línea 235: Arco cupido derecho no definido 70%'
    ),
    'ac_d_on': ThresholdRule(
        diagnosis_name='ac_d_on',
        threshold=0.40,  # NewFeature.md línea 237: "on (40%)"
        rule_type='minimum',
        description='NewFeature.md línea 237: Arco cupido derecho marcado 40%'
    ),
    'ac_d_pc': ThresholdRule(
        diagnosis_name='ac_d_pc',
        threshold=0.10,  # NewFeature.md línea 239: "pc (10%)"
        rule_type='minimum',
        description='NewFeature.md línea 239: Arco cupido derecho triangular 10%'
    ),
    'ac_i_nd': ThresholdRule(
        diagnosis_name='ac_i_nd',
        threshold=0.70,  # NewFeature.md línea 235: "nd (70%)"
        rule_type='minimum',
        description='NewFeature.md línea 235: Arco cupido izquierdo no definido 70%'
    ),
    'ac_i_on': ThresholdRule(
        diagnosis_name='ac_i_on',
        threshold=0.40,  # NewFeature.md línea 237: "on (40%)"
        rule_type='minimum',
        description='NewFeature.md línea 237: Arco cupido izquierdo marcado 40%'
    ),
    'ac_i_pc': ThresholdRule(
        diagnosis_name='ac_i_pc',
        threshold=0.10,  # NewFeature.md línea 239: "pc (10%)"
        rule_type='minimum',
        description='NewFeature.md línea 239: Arco cupido izquierdo triangular 10%'
    ),
}

# ============================================================================
# NewFeature.md línea 73: Regla de 15 puntos de diferencia
# ============================================================================
# "si no existe umbral, se toma el de mayor porcentaje tenga mínimo
#  15 puntos de porcentaje mayor a los demás"
MORFOLOGICO_MINIMUM_DIFFERENCE = 0.15  # 15 puntos de porcentaje


# ============================================================================
# NewFeature.md: CONFIGURATION DICT - Single Source of Truth
# ============================================================================
# Diccionario principal que mapea cada ModuleType a sus reglas de umbral.
# Este es el único lugar donde se definen todos los umbrales del sistema.
# ============================================================================

# ============================================================================
# NewFeature.md: Modulo Perfil morfologico (líneas 271-373)
# ============================================================================
# Umbrales para diagnósticos de perfil morfológico.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 271-373
#
# IMPORTANTE (línea 273):
# "En este modulo solo un resultado para izq y der, excepto por lobulo y mandibula"
# ============================================================================

PROFILE_MORFOLOGICO_RULES = {
    # ========================================================================
    # NewFeature.md líneas 291-295: Lóbulo (lobulo_izquierdo, lobulo_derecho)
    # ========================================================================
    'lobulo_izquierdo_pegado': ThresholdRule(
        diagnosis_name='lobulo_izquierdo_pegado',
        threshold=0.60,  # NewFeature.md línea 293: "lobulo_pegado (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 293: Lóbulo izquierdo pegado 60%'
    ),
    'lobulo_izquierdo_despegado': ThresholdRule(
        diagnosis_name='lobulo_izquierdo_despegado',
        threshold=0.60,  # NewFeature.md línea 295: "lobulo_despegado (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 295: Lóbulo izquierdo despegado 60%'
    ),
    'lobulo_izquierdo_hacia_adelante': ThresholdRule(
        diagnosis_name='lobulo_izquierdo_hacia_adelante',
        threshold=0.20,  # NewFeature.md línea 297: "lobulo_hacia_adelante (20%)"
        rule_type='minimum',
        description='NewFeature.md línea 297: Lóbulo izquierdo hacia adelante 20%'
    ),
    'lobulo_derecho_pegado': ThresholdRule(
        diagnosis_name='lobulo_derecho_pegado',
        threshold=0.60,  # NewFeature.md línea 293: "lobulo_pegado (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 293: Lóbulo derecho pegado 60%'
    ),
    'lobulo_derecho_despegado': ThresholdRule(
        diagnosis_name='lobulo_derecho_despegado',
        threshold=0.60,  # NewFeature.md línea 295: "lobulo_despegado (60%)"
        rule_type='minimum',
        description='NewFeature.md línea 295: Lóbulo derecho despegado 60%'
    ),
    'lobulo_derecho_hacia_adelante': ThresholdRule(
        diagnosis_name='lobulo_derecho_hacia_adelante',
        threshold=0.20,  # NewFeature.md línea 297: "lobulo_hacia_adelante (20%)"
        rule_type='minimum',
        description='NewFeature.md línea 297: Lóbulo derecho hacia adelante 20%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 313-321: Submentón (submenton)
    # ========================================================================
    # Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 313-321
    # "Se toma el diagnostico con mayor certeza si arriba de 45%"
    # ========================================================================
    'submenton_visible': ThresholdRule(
        diagnosis_name='submenton_visible',
        threshold=0.45,  # NewFeature.md línea 315: "si arriba de 45%"
        rule_type='minimum',
        description='NewFeature.md línea 315: Submentón visible 45%'
    ),
    'submenton_no_visible': ThresholdRule(
        diagnosis_name='submenton_no_visible',
        threshold=0.45,  # NewFeature.md línea 315: "si arriba de 45%"
        rule_type='minimum',
        description='NewFeature.md línea 315: Submentón no visible 45%'
    ),
    
    # ========================================================================
    # NewFeature.md líneas 331-347: Frente morfológico
    # ========================================================================
    'f_rd_incl_derecha': ThresholdRule(
        diagnosis_name='f_rd_incl_derecha',
        threshold=0.27,  # NewFeature.md línea 337: "f_rd_incl (mayor a 27%)"
        rule_type='minimum',
        description='NewFeature.md línea 337: Frente derecha redondeada inclinada >27%'
    ),
    'f_rd_incl_derecha_alta': ThresholdRule(
        diagnosis_name='f_rd_incl_derecha_alta',
        threshold=0.55,  # NewFeature.md línea 339: "(mayor a 55%)"
        rule_type='minimum',
        description='NewFeature.md línea 339: Frente derecha redondeada inclinada >55%'
    ),
    'f_pl_incl_derecha': ThresholdRule(
        diagnosis_name='f_pl_incl_derecha',
        threshold=0.27,  # NewFeature.md línea 341: "f_pl_incl (27%)"
        rule_type='minimum',
        description='NewFeature.md línea 341: Frente derecha plana inclinada 27%'
    ),
    'fr_vert_derecha': ThresholdRule(
        diagnosis_name='fr_vert_derecha',
        threshold=0.20,  # NewFeature.md línea 343: "fr_vert (20% perfil derecho)"
        rule_type='minimum',
        description='NewFeature.md línea 343: Frente vertical derecha 20%'
    ),
    'ab_t_inf_derecha': ThresholdRule(
        diagnosis_name='ab_t_inf_derecha',
        threshold=0.25,  # NewFeature.md línea 345: "ab_t_inf (25% derecho)"
        rule_type='minimum',
        description='NewFeature.md línea 345: Abultamiento tercio inferior derecho 25%'
    ),
    'f_rd_incl_izquierda': ThresholdRule(
        diagnosis_name='f_rd_incl_izquierda',
        threshold=0.27,  # NewFeature.md línea 337: "f_rd_incl (mayor a 27%)"
        rule_type='minimum',
        description='NewFeature.md línea 337: Frente izquierda redondeada inclinada >27%'
    ),
    'fr_vert_izquierda': ThresholdRule(
        diagnosis_name='fr_vert_izquierda',
        threshold=0.19,  # NewFeature.md línea 343: "fr_vert (19% perfil izquierdo)"
        rule_type='minimum',
        description='NewFeature.md línea 343: Frente vertical izquierda 19%'
    ),
    'ab_t_inf_izquierda': ThresholdRule(
        diagnosis_name='ab_t_inf_izquierda',
        threshold=0.21,  # NewFeature.md línea 345: "ab_t_inf (21% perfil izq)"
        rule_type='minimum',
        description='NewFeature.md línea 345: Abultamiento tercio inferior izquierda 21%'
    ),
}

# ============================================================================
# NewFeature.md: Modulo Perfil antropométrico (líneas 374-428)
# ============================================================================
# Umbrales y reglas para perfil antropométrico.
# Referencia: /home/mitza/proyectos/SOUL-GATE/NewFeature.md líneas 374-428
# ============================================================================

PROFILE_ANTROPOMETRICO_RULES = {
    # Nota: La mayoría de las reglas de perfil antropométrico no tienen umbrales
    # explícitos, sino reglas de coincidencia (ver RuleEngine)
    # Aquí solo incluimos las que tienen umbrales específicos
}


THRESHOLD_CONFIG = {
    ModuleType.ESPEJO_FRENTE: ESPEJO_FRENTE_RULES,
    ModuleType.ESPEJO_ROSTRO: ESPEJO_ROSTRO_RULES,
    ModuleType.FRONTAL_MORFOLOGICO: FRONTAL_MORFOLOGICO_RULES,
    ModuleType.PROFILE_MORFOLOGICO: PROFILE_MORFOLOGICO_RULES,
    ModuleType.PROFILE_ANTROPOMETRICO: PROFILE_ANTROPOMETRICO_RULES,
    # Nota: FRONTAL_ANTROPOMETRICO no tiene umbrales específicos (línea 77)
}

