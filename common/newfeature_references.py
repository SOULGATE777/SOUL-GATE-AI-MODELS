# ============================================================================
# NewFeature.md References - Mapeo completo de diagnósticos a líneas y texto
# ============================================================================
# Este archivo mapea cada diagnóstico de los servicios ML a las líneas exactas
# y texto de /home/mitza/proyectos/SOUL-GATE/NewFeature.md que lo definen.
#
# Estructura:
# {
#     "diagnosis_name": {
#         "lines": "31-31",  # Líneas donde aparece en NewFeature.md
#         "text": "Texto completo de la definición...",
#         "threshold_line": "23",  # Línea donde se define el umbral
#         "threshold_text": "Venus Corazon (mayor a 40%)",
#         "threshold_value": 0.40,  # Valor numérico del umbral
#         "module": "espejo",  # Módulo al que pertenece
#         "category": "rostro",  # Categoría dentro del módulo
#         "personality_or_temperament": "personalidad"  # o "temperamento"
#     }
# }
# ============================================================================

from typing import Dict, Any, List, Optional

# ============================================================================
# MÓDULO: ESPEJO - ROSTRO
# NewFeature.md líneas 13-54
# ============================================================================

ESPEJO_ROSTRO_REFERENCES: Dict[str, Dict[str, Any]] = {
    # Umbral general
    "_general": {
        "lines": "17-17",
        "text": "Si predicción de rostro es mayor al 18% se toma como diagnostico certero, de lo contario, se omite.",
        "threshold_line": "17",
        "threshold_text": "predicción de rostro es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad"
    },
    
    # Excepciones
    "venus_corazon": {
        "lines": "31-31",
        "text": "venus corazon: Les encanta amar y saberse amados, especialmente cuando reciben adulación, adoración y una explicación clara de por qué se les aprecia. La armonía y lo estético es muy importante para ellos. Son vivaces, valientes e ingeniosos, con una inteligencia práctica que les permite organizar las cosas de manera que todos se sientan cómodos. Suelen hablar como una forma de liberar la tensión emocional, y a menudo se preocupan por quienes no pueden defenderse por sí mismos. Afinidad por los niños, animales, y/o cosas tiernas y comúnmente por las causas sociales.",
        "threshold_line": "23",
        "threshold_text": "Venus Corazon (mayor a 40%)",
        "threshold_value": 0.40,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad",
        "is_exception": True
    },
    
    "pluton_hexagonal": {
        "lines": "37-37",
        "text": "pluton_hexagonal: Son personas muy orientadas hacia lo físico, por lo que necesitan una vía de escape para su energía. Les atrae el dinero y suelen aprender con facilidad cómo generarlo, lo que los vuelve buenos en los negocios rápidamente. Sin embargo, en su afán por alcanzar sus metas, pueden mostrar cierta amoralidad en los métodos que emplean. Los Plutones son creativos, muy organizados y afectuosos. Salvo si estan pasando por una depresión, suelen ser positivos y sociables. Poseen una lealtad excepcional y destacan como cuidadores, ofreciendo apoyo genuino a aquellas personas cercanas.",
        "threshold_line": "23",
        "threshold_text": "Pluton Hexagonal (cuyo umbral es mayor a 7%)",
        "threshold_value": 0.07,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad",
        "is_exception": True
    },
    
    "sol_neptuno_combined": {
        "lines": "33-33",
        "text": "sol_neptuno_combined y proporción Facial y proporcion menor a 1.17: Estas personas necesitan mucho espacio y tiempo para sí mismas. Son exigentes, especialmente al elegir a sus amigos íntimos, y poseen una inclinación natural hacia el liderazgo. Carismáticos pero distantes, aprenden con facilidad, aunque no suelen compartir sus ideas a menos que se les pregunte—y, aun así, suelen saber más de lo que expresan.Tienden a dedicar su vida a una causa u objetivo, y encuentran satisfacción en ello sin depender de la aprobación externa. Son analíticos y creativos, impulsados a avanzar con convicción.",
        "threshold_line": "17",
        "threshold_text": "predicción de rostro es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad"
    },
    
    "marte_tierra_rectangulo": {
        "lines": "35-39",
        "text": "marte_tierra_rectangulo: Muy orientados a la acción, físicos y probablemente impacientes. Pueden variar según proporción facial.",
        "threshold_line": "17",
        "threshold_text": "predicción de rostro es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad"
    },
    
    "saturno_trapezoide_base_angosta": {
        "lines": "41-43",
        "text": "Saturno_trapezoide_base_angosta: Saben con claridad qué quieren lograr. Contemplativoscon marcada conciencia del tiempo. Pensadores filosóficos profundos y lógicos.",
        "threshold_line": "17",
        "threshold_text": "predicción de rostro es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad"
    },
    
    "luna_jupiter_combined": {
        "lines": "45-47",
        "text": "luna_jupiter_combined: Estables, decididos y profundamente leales. Pensadores con visión amplia. Varía según proporción facial.",
        "threshold_line": "17",
        "threshold_text": "predicción de rostro es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad"
    },
    
    "mercurio_triangular": {
        "lines": "51-53",
        "text": "mercurio_triangular: rápidos de mente, volátil, talentoso y muy comunicativo. Normalmente divertido, encantador. Varía según proporción facial.",
        "threshold_line": "17",
        "threshold_text": "predicción de rostro es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "rostro",
        "personality_or_temperament": "personalidad"
    }
}

# ============================================================================
# MÓDULO: ESPEJO - FRENTE
# NewFeature.md líneas 55-69
# ============================================================================

ESPEJO_FRENTE_REFERENCES: Dict[str, Dict[str, Any]] = {
    "_general": {
        "lines": "19-19",
        "text": "Si predicción de frente es mayor al 18% se toma como diagnostico certero, de lo contario, se omite.",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    },
    
    "solar_lunar_combined": {
        "lines": "57-59",
        "text": "solar_lunar_combined: Persona emotiva, exigentes, carismática. Varía según proporción frente.",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    },
    
    "mercurio_triangulo": {
        "lines": "61-61",
        "text": "mercurio_triangulo: persona creativa",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    },
    
    "marte_rectangular": {
        "lines": "63-63",
        "text": "marte rectangular: Persona orientada a accionar, hacer o resultados",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    },
    
    "jupiter_amplio_base_ancha": {
        "lines": "65-65",
        "text": "jupiter_amplio_base_ancha: persona capaz de ver un panorama amplio de cómo varios factores interactúan pero puede caer en enjuiciar demasiado.",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    },
    
    "neptuno_combined": {
        "lines": "67-67",
        "text": "neptuno_combined: Persona con gran capacidad de análisis, intuitiva y muy pensante que se presione mucho para lograr objetivos, capacidad de pensamiento abstracto pero puede perderse en esas abstracciones y no ver claramente.",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    },
    
    "venus_corazon_o_trapezoide_angosto": {
        "lines": "69-69",
        "text": "Venus Corazon O Trapezoide Angosto: persona interesada en bienestar de otros, charismatica",
        "threshold_line": "19",
        "threshold_text": "predicción de frente es mayor al 18%",
        "threshold_value": 0.18,
        "module": "espejo",
        "category": "frente",
        "personality_or_temperament": "personalidad"
    }
}

# ============================================================================
# MÓDULO: FRONTAL ANTROPOMÉTRICO
# NewFeature.md líneas 75-118
# ============================================================================

FRONTAL_ANTROPOMETRICO_REFERENCES: Dict[str, Dict[str, Any]] = {
    "_general": {
        "lines": "77-77",
        "text": "Todo diagnóstico de frontal antropométrico se toma exactamente como está (asegurandote de distinguir entre lado izquierdo o derecho si no especifica)",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "general",
        "personality_or_temperament": "personalidad"
    },
    
    # Tamaño de ojo
    "ojo_grande": {
        "lines": "89-89",
        "text": "ojo grande: persona emotiva que se puede llegar a abrumar con facilidad",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "tamaño_ojo",
        "personality_or_temperament": "personalidad"
    },
    
    "ojo_mediano": {
        "lines": "91-91",
        "text": "ojo mediano: no se menciona",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "tamaño_ojo",
        "personality_or_temperament": "personalidad"
    },
    
    "ojo_pequeño": {
        "lines": "93-93",
        "text": "ojo pequeño (ambos ojo derecho e ojo izquierdo): persona lógica que no se abruma con facilidad",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "tamaño_ojo",
        "personality_or_temperament": "personalidad"
    },
    
    # Tamaño de boca
    "boca_grande": {
        "lines": "101-101",
        "text": "boca grande: persona que suele hablar mucho, no suele limitarse mucho o más impulsiva",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "tamaño_boca",
        "personality_or_temperament": "personalidad"
    },
    
    "boca_promedio": {
        "lines": "103-103",
        "text": "Boca promedio: no se menciona",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "tamaño_boca",
        "personality_or_temperament": "personalidad"
    },
    
    "boca_pequeña": {
        "lines": "105-105",
        "text": "Boca pequeña: persona que se controla o limita, no habla tanto",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "tamaño_boca",
        "personality_or_temperament": "personalidad"
    },
    
    # Análisis de área facial
    "cara_interna_promedio": {
        "lines": "113-113",
        "text": "cara interna promedio: no se menciona",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "area_facial",
        "personality_or_temperament": "personalidad"
    },
    
    "cara_interna_pequeña": {
        "lines": "115-115",
        "text": "cara interna pequeña: persona no muy afectada por eventos externos, ya sea porque logra racionalizarlos, se siente con las herramientas para manejarlos, o está desapegado emocionalmente.",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "area_facial",
        "personality_or_temperament": "personalidad"
    },
    
    "cara_interna_grande": {
        "lines": "117-117",
        "text": "cara interna grande: persona que vive el momento muy intensamente, mas afectada por eventos externos",
        "threshold_line": "77",
        "threshold_text": "se toma exactamente como está",
        "threshold_value": None,
        "module": "frontal_antropometrico",
        "category": "area_facial",
        "personality_or_temperament": "personalidad"
    }
}

# ============================================================================
# MÓDULO: FRONTAL MORFOLÓGICO
# NewFeature.md líneas 119-270
# ============================================================================

FRONTAL_MORFOLOGICO_REFERENCES: Dict[str, Dict[str, Any]] = {
    "_general": {
        "lines": "73-73",
        "text": "tenga mínimo 15 puntos de porcentaje mayor a los demás posibles diagnósticos, de no cumplir con alguno de los anteriores criterios se omite",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "general",
        "personality_or_temperament": "personalidad"
    },
    
    # Cejas (líneas 123-132)
    "ceja_curva": {
        "lines": "127-127",
        "text": "cV (50%): ceja curva - mente abierta, adaptabilidad, posible dificultad poniendo límites",
        "threshold_line": "127",
        "threshold_text": "cV (50%)",
        "threshold_value": 0.50,
        "module": "frontal_morfologico",
        "category": "cejas",
        "personality_or_temperament": "personalidad"
    },
    
    "ceja_inclinada": {
        "lines": "129-129",
        "text": "el (50%): ceja inclinada o elevada - combativa o crítica, impulso a la acción, hiperreactiva, pensamiento rápido y creativo, impaciencia",
        "threshold_line": "129",
        "threshold_text": "el (50%)",
        "threshold_value": 0.50,
        "module": "frontal_morfologico",
        "category": "cejas",
        "personality_or_temperament": "personalidad"
    },
    
    "ceja_recta": {
        "lines": "131-131",
        "text": "rc (50%): ceja de forma recta - Enfocada, determinada, más tendente a la terquedad, mas probable de poner limites",
        "threshold_line": "131",
        "threshold_text": "rc (50%)",
        "threshold_value": 0.50,
        "module": "frontal_morfologico",
        "category": "cejas",
        "personality_or_temperament": "personalidad"
    },
    
    # Entrecejo (líneas 133-141)
    "entrecejo_normal": {
        "lines": "137-137",
        "text": "normal: normal - no se menciona",
        "threshold_line": "137",
        "threshold_text": "normal",
        "threshold_value": None,
        "module": "frontal_morfologico",
        "category": "entrecejo",
        "personality_or_temperament": "personalidad"
    },
    
    "uniceja": {
        "lines": "139-139",
        "text": "uniceja (60%): uniceja - desconección de sus emociones o intuición",
        "threshold_line": "139",
        "threshold_text": "uniceja (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "entrecejo",
        "personality_or_temperament": "personalidad"
    },
    
    "lineas_verticales": {
        "lines": "141-141",
        "text": "lineas_verticales (60%): líneas verticales - pasa mucho tiempo en contemplación o siente ira con muy frecuentemente ya sea que la exprese o reprima",
        "threshold_line": "141",
        "threshold_text": "lineas_verticales (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "entrecejo",
        "personality_or_temperament": "personalidad"
    },
    
    # Párpado (líneas 143-149)
    "ptosis": {
        "lines": "147-147",
        "text": "ptosis (60%): ptosis palpebral - persona más privada y suele respetar privacidad ajena",
        "threshold_line": "147",
        "threshold_text": "ptosis (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "parpado",
        "personality_or_temperament": "personalidad"
    },
    
    "pliegue": {
        "lines": "149-149",
        "text": "pliegue (60%): pliegue palpebral - no se menciona",
        "threshold_line": "149",
        "threshold_text": "pliegue (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "parpado",
        "personality_or_temperament": "personalidad"
    },
    
    # Ojo (líneas 151-163)
    "ojo_circular": {
        "lines": "155-155",
        "text": "crl: ojo circular - mente abierta, emotiva, expresiva",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "ojo",
        "personality_or_temperament": "personalidad"
    },
    
    "ojo_almendrado": {
        "lines": "157-157",
        "text": "al (22%): ojo almendrado - persona más lógica que emotiva",
        "threshold_line": "157",
        "threshold_text": "al (22%)",
        "threshold_value": 0.22,
        "module": "frontal_morfologico",
        "category": "ojo",
        "personality_or_temperament": "personalidad"
    },
    
    "ojo_fruncido": {
        "lines": "159-159",
        "text": "fr (35%): ojo fruncido - pragmático, práctico, desconfiado",
        "threshold_line": "159",
        "threshold_text": "fr (35%)",
        "threshold_value": 0.35,
        "module": "frontal_morfologico",
        "category": "ojo",
        "personality_or_temperament": "personalidad"
    },
    
    "ojo_media_luna_arriba": {
        "lines": "161-161",
        "text": "md (70%): ojo media luna hacia arriba - persona strategica, soñadora",
        "threshold_line": "161",
        "threshold_text": "md (70%)",
        "threshold_value": 0.70,
        "module": "frontal_morfologico",
        "category": "ojo",
        "personality_or_temperament": "personalidad"
    },
    
    "ojo_media_luna_abajo": {
        "lines": "163-163",
        "text": "md_a (22%): ojo media luna hacia abajo - persona dispuesta a luchar por ideales",
        "threshold_line": "163",
        "threshold_text": "md_a (22%)",
        "threshold_value": 0.22,
        "module": "frontal_morfologico",
        "category": "ojo",
        "personality_or_temperament": "personalidad"
    },
    
    # Oído (líneas 165-175)
    "oido_tercio_superior_salido": {
        "lines": "169-169",
        "text": "sp_sl (30%): tercio superior salido - persona apegada pero tendente a pesar fueras de las normas, buscando temas específico de su interés",
        "threshold_line": "169",
        "threshold_text": "sp_sl (30%)",
        "threshold_value": 0.30,
        "module": "frontal_morfologico",
        "category": "oido",
        "personality_or_temperament": "personalidad"
    },
    
    "oido_salido": {
        "lines": "171-171",
        "text": "sl (33%): persona desapegada y tendentes a pensar fuera de las normas",
        "threshold_line": "171",
        "threshold_text": "sl (33%)",
        "threshold_value": 0.33,
        "module": "frontal_morfologico",
        "category": "oido",
        "personality_or_temperament": "personalidad"
    },
    
    "oido_pegado": {
        "lines": "173-173",
        "text": "pg (25%): pegados - persona que suele apegarse a protocolos y normas, suele ser bueno para escuchar",
        "threshold_line": "173",
        "threshold_text": "pg (25%)",
        "threshold_value": 0.25,
        "module": "frontal_morfologico",
        "category": "oido",
        "personality_or_temperament": "personalidad"
    },
    
    "oido_promedio": {
        "lines": "175-175",
        "text": "pm (25%): promedio - capaz de apegarse a normas o salirse de ellas",
        "threshold_line": "175",
        "threshold_text": "pm (25%)",
        "threshold_value": 0.25,
        "module": "frontal_morfologico",
        "category": "oido",
        "personality_or_temperament": "personalidad"
    },
    
    # Nariz - Grosor puente (líneas 177-185)
    "nariz_normal": {
        "lines": "181-181",
        "text": "nrml (65%): normal - no se menciona",
        "threshold_line": "181",
        "threshold_text": "nrml (65%)",
        "threshold_value": 0.65,
        "module": "frontal_morfologico",
        "category": "nariz_grosor",
        "personality_or_temperament": "personalidad"
    },
    
    "nariz_grueso": {
        "lines": "183-183",
        "text": "grueso (22%): grueso - mucha energia y resistencia física",
        "threshold_line": "183",
        "threshold_text": "grueso (22%)",
        "threshold_value": 0.22,
        "module": "frontal_morfologico",
        "category": "nariz_grosor",
        "personality_or_temperament": "personalidad"
    },
    
    "nariz_delgada": {
        "lines": "185-185",
        "text": "delgada (14%): delgada - persona creativa e hiperreactiva, tendencia a ansiedad",
        "threshold_line": "185",
        "threshold_text": "delgada (14%)",
        "threshold_value": 0.14,
        "module": "frontal_morfologico",
        "category": "nariz_grosor",
        "personality_or_temperament": "personalidad"
    },
    
    # Punta de nariz (líneas 187-193)
    "punta_nariz_redondeada": {
        "lines": "189-189",
        "text": "rd (50%): socialmente consciente",
        "threshold_line": "189",
        "threshold_text": "rd (50%)",
        "threshold_value": 0.50,
        "module": "frontal_morfologico",
        "category": "punta_nariz",
        "personality_or_temperament": "personalidad"
    },
    
    "punta_nariz_indiferenciada": {
        "lines": "191-191",
        "text": "i: indiferenciada - no se menciona",
        "threshold_line": "191",
        "threshold_text": "i",
        "threshold_value": None,
        "module": "frontal_morfologico",
        "category": "punta_nariz",
        "personality_or_temperament": "personalidad"
    },
    
    "punta_nariz_puntiaguda": {
        "lines": "193-193",
        "text": "pn (60%): puntiagudo - persona enfocada en conseguir lo que quiere",
        "threshold_line": "193",
        "threshold_text": "pn (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "punta_nariz",
        "personality_or_temperament": "personalidad"
    },
    
    # Pómulo (líneas 197-205)
    "pomulo_salido": {
        "lines": "201-201",
        "text": "salido: salido - persona resiliente, probablemente paciente, estandartes altos, gran necesidad de conexión",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "pomulo",
        "personality_or_temperament": "personalidad"
    },
    
    "pomulo_promedio": {
        "lines": "203-203",
        "text": "pm (60%): promedio - no se menciona",
        "threshold_line": "203",
        "threshold_text": "pm (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "pomulo",
        "personality_or_temperament": "personalidad"
    },
    
    "pomulo_plano": {
        "lines": "205-205",
        "text": "pl (60%): plano - probablemente impaciente",
        "threshold_line": "205",
        "threshold_text": "pl (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "pomulo",
        "personality_or_temperament": "personalidad"
    },
    
    # Cachete (líneas 207-217)
    "cachete_lleno": {
        "lines": "211-211",
        "text": "ll (60%): lleno - persona segura de sí pero que siente necesidad de tener una reserva",
        "threshold_line": "211",
        "threshold_text": "ll (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "cachete",
        "personality_or_temperament": "personalidad"
    },
    
    "cachete_plano": {
        "lines": "213-213",
        "text": "pl (80%): planos - no se menciona",
        "threshold_line": "213",
        "threshold_text": "pl (80%)",
        "threshold_value": 0.80,
        "module": "frontal_morfologico",
        "category": "cachete",
        "personality_or_temperament": "personalidad"
    },
    
    "cachete_hundido": {
        "lines": "215-215",
        "text": "hn (60%): hundidos - persona confrontativa",
        "threshold_line": "215",
        "threshold_text": "hn (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "cachete",
        "personality_or_temperament": "personalidad"
    },
    
    "lineas_sonrisa": {
        "lines": "217-217",
        "text": "lineas_sonriza (80%): - persona que conoce su propósito de vida",
        "threshold_line": "217",
        "threshold_text": "lineas_sonriza (80%)",
        "threshold_value": 0.80,
        "module": "frontal_morfologico",
        "category": "cachete",
        "personality_or_temperament": "personalidad"
    },
    
    # Forma de boca (líneas 219-229)
    "boca_lunar": {
        "lines": "223-223",
        "text": "lunar (60%)- persona que se comunica de forma emotiva, difusa, o poetica probablemente alta necesidad emocional",
        "threshold_line": "223",
        "threshold_text": "lunar (60%)",
        "threshold_value": 0.60,
        "module": "frontal_morfologico",
        "category": "forma_boca",
        "personality_or_temperament": "personalidad"
    },
    
    "boca_solar": {
        "lines": "225-225",
        "text": "solar (30%) - persona que se comunica de forma logica y analitica",
        "threshold_line": "225",
        "threshold_text": "solar (30%)",
        "threshold_value": 0.30,
        "module": "frontal_morfologico",
        "category": "forma_boca",
        "personality_or_temperament": "personalidad"
    },
    
    "boca_mercurial": {
        "lines": "227-227",
        "text": "mercurial (50%) - persona creativa y buena para comunicarse pero que no suele compartir sus sentimientos verdaderos a primera instancia.",
        "threshold_line": "227",
        "threshold_text": "mercurial (50%)",
        "threshold_value": 0.50,
        "module": "frontal_morfologico",
        "category": "forma_boca",
        "personality_or_temperament": "personalidad"
    },
    
    "boca_pursed": {
        "lines": "229-229",
        "text": "pursed (30%) - persona que se exige mucho a sí mismo y probablemente a los demás aunque lo exprese o no.",
        "threshold_line": "229",
        "threshold_text": "pursed (30%)",
        "threshold_value": 0.30,
        "module": "frontal_morfologico",
        "category": "forma_boca",
        "personality_or_temperament": "personalidad"
    },
    
    # Arco de cupido (líneas 231-239)
    "arco_cupido_no_definido": {
        "lines": "235-235",
        "text": "nd (70%): no definido - no se menciona",
        "threshold_line": "235",
        "threshold_text": "nd (70%)",
        "threshold_value": 0.70,
        "module": "frontal_morfologico",
        "category": "arco_cupido",
        "personality_or_temperament": "personalidad"
    },
    
    "arco_cupido_marcado": {
        "lines": "237-237",
        "text": "on (40%): arco de cupido marcado - persona sensual que disfruta armonia, coneccion, atencion",
        "threshold_line": "237",
        "threshold_text": "on (40%)",
        "threshold_value": 0.40,
        "module": "frontal_morfologico",
        "category": "arco_cupido",
        "personality_or_temperament": "personalidad"
    },
    
    "arco_cupido_triangular": {
        "lines": "239-239",
        "text": "pc (10%): arco de cupido triangular - persona creativa al comunicar",
        "threshold_line": "239",
        "threshold_text": "pc (10%)",
        "threshold_value": 0.10,
        "module": "frontal_morfologico",
        "category": "arco_cupido",
        "personality_or_temperament": "personalidad"
    },
    
    # Tercios faciales (líneas 241-259)
    "tercio_medio_largo": {
        "lines": "245-245",
        "text": "tercio medio largo: persona social, empática, con mucha necesidad de conecciones y socializar",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "tercios_faciales",
        "personality_or_temperament": "personalidad"
    },
    
    "tercio_medio_corto": {
        "lines": "247-247",
        "text": "tercio medio corto: persona no muy empática, sin mucha necesidad de socializar",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "tercios_faciales",
        "personality_or_temperament": "personalidad"
    },
    
    "tercio_medio_standard": {
        "lines": "249-249",
        "text": "tercio medio standard: no se menciona",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "tercios_faciales",
        "personality_or_temperament": "personalidad"
    },
    
    "tercio_inferior_corto": {
        "lines": "255-255",
        "text": "Tercio inferior corto: persona que no le da mucha importancia a lo material o físico",
        "threshold_line": "73",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "tercios_faciales",
        "personality_or_temperament": "personalidad"
    },
    
    "tercio_inferior_largo": {
        "lines": "257-257",
        "text": "Tercio inferior largo: persona que le pone mucha importancia a lo material o físico.",
        "threshold_line": "257",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "tercios_faciales",
        "personality_or_temperament": "personalidad"
    },
    
    "tercio_inferior_standard": {
        "lines": "259-259",
        "text": "Tercio inferior standard: no se menciona",
        "threshold_line": "259",
        "threshold_text": "mínimo 15 puntos de porcentaje mayor",
        "threshold_value": 0.15,
        "module": "frontal_morfologico",
        "category": "tercios_faciales",
        "personality_or_temperament": "personalidad"
    }
}

# ============================================================================
# MÓDULO: PERFIL MORFOLÓGICO
# NewFeature.md líneas 271-373
# ============================================================================

PROFILE_MORFOLOGICO_REFERENCES: Dict[str, Dict[str, Any]] = {
    # Dorso de nariz (líneas 275-287)
    "nariz_convexa": {
        "lines": "283-283",
        "text": "convexa - convexa: persona sensible, creativa e hiperreactiva",
        "threshold_line": "277",
        "threshold_text": "tomar el diagnostico con mayor porcentaje de certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "dorso_nariz",
        "personality_or_temperament": "temperamento"
    },
    
    "nariz_concava": {
        "lines": "285-285",
        "text": "concava - introspectiva y no tan fácilmente afectada por eventos externos",
        "threshold_line": "277",
        "threshold_text": "tomar el diagnostico con mayor porcentaje de certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "dorso_nariz",
        "personality_or_temperament": "temperamento"
    },
    
    "nariz_recta": {
        "lines": "287-287",
        "text": "recta - enfocado en conseguir lo que quiere",
        "threshold_line": "277",
        "threshold_text": "tomar el diagnostico con mayor porcentaje de certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "dorso_nariz",
        "personality_or_temperament": "temperamento"
    },
    
    # Lóbulo (líneas 289-297)
    "lobulo_pegado": {
        "lines": "293-293",
        "text": "lobulo_pegado (60%): persona que requiere apoyo para iniciar proyectos",
        "threshold_line": "293",
        "threshold_text": "lobulo_pegado (60%)",
        "threshold_value": 0.60,
        "module": "profile_morfologico",
        "category": "lobulo",
        "personality_or_temperament": "personalidad"
    },
    
    "lobulo_despegado": {
        "lines": "295-295",
        "text": "lobulo_despegado (60%): persona se siente cómoda realizando proyectos sin mucho apoyo",
        "threshold_line": "295",
        "threshold_text": "lobulo_despegado (60%)",
        "threshold_value": 0.60,
        "module": "profile_morfologico",
        "category": "lobulo",
        "personality_or_temperament": "personalidad"
    },
    
    "lobulo_hacia_adelante": {
        "lines": "297-297",
        "text": "lobulo_hacia_adelante (20%)",
        "threshold_line": "297",
        "threshold_text": "lobulo_hacia_adelante (20%)",
        "threshold_value": 0.20,
        "module": "profile_morfologico",
        "category": "lobulo",
        "personality_or_temperament": "personalidad"
    },
    
    # Mandíbula (líneas 299-311)
    "mandibula_nerviosa": {
        "lines": "305-305",
        "text": "m_crt_n: mandibula nerviosa - persona creativa que con su sistema nervioso bastante estimulado, probablemente tiene dificultad para ver proyectos hasta el final",
        "threshold_line": "301",
        "threshold_text": "Se toma diagnostico con mayor certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    "mandibula_biliosa": {
        "lines": "307-307",
        "text": "m_m_vrt_b: mandibula biliosa",
        "threshold_line": "301",
        "threshold_text": "Se toma diagnostico con mayor certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    "mandibula_sanguinea": {
        "lines": "309-309",
        "text": "m_lrg_vert_s: mandibula sanguinea",
        "threshold_line": "301",
        "threshold_text": "Se toma diagnostico con mayor certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    "mandibula_linfatica": {
        "lines": "311-311",
        "text": "m_linf: mandibula linfatica - esta characteristica templa al individuo un poco de sus caracteristicas internas y lo hace un poco mas resistente y empatico.",
        "threshold_line": "301",
        "threshold_text": "Se toma diagnostico con mayor certeza",
        "threshold_value": None,
        "module": "profile_morfologico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    # Submentón (líneas 313-321)
    "submenton_no_visible": {
        "lines": "319-319",
        "text": "no_visible: no se menciona",
        "threshold_line": "315",
        "threshold_text": "Se toma el diagnostico con mayor certeza si arriba de 45%",
        "threshold_value": 0.45,
        "module": "profile_morfologico",
        "category": "submenton",
        "personality_or_temperament": "temperamento"
    },
    
    "submenton_visible": {
        "lines": "321-321",
        "text": "visible: buena memoria, tendencia a retener o dificultad para soltar",
        "threshold_line": "315",
        "threshold_text": "Se toma el diagnostico con mayor certeza si arriba de 45%",
        "threshold_value": 0.45,
        "module": "profile_morfologico",
        "category": "submenton",
        "personality_or_temperament": "temperamento"
    },
    
    # Frente morfológico (líneas 331-347)
    "frente_redondeada_inclinada": {
        "lines": "337-339",
        "text": "f_rd_incl (mayor a 27%): frente redondeada y/o inclinada - persona pasional/emotiva que puede requerir novedad",
        "threshold_line": "337",
        "threshold_text": "f_rd_incl (mayor a 27%)",
        "threshold_value": 0.27,
        "module": "profile_morfologico",
        "category": "frente",
        "personality_or_temperament": "temperamento"
    },
    
    "frente_plana_inclinada": {
        "lines": "341-341",
        "text": "f_pl_incl (27%): frente plana - persona analitica, desapegada que suele requiere novedad o sin gran capacidad de atención prolongada.",
        "threshold_line": "341",
        "threshold_text": "f_pl_incl (27%)",
        "threshold_value": 0.27,
        "module": "profile_morfologico",
        "category": "frente",
        "personality_or_temperament": "temperamento"
    },
    
    "frente_vertical": {
        "lines": "343-343",
        "text": "fr_vert (20% perfil derecho, 19% perfil izquierdo): frente vertical - persona analitica, capaz de enfocarse en las cosas por largos periodos de tiempo, determinada que puede llegar a terquedad.",
        "threshold_line": "343",
        "threshold_text": "fr_vert (20% perfil derecho, 19% perfil izquierdo)",
        "threshold_value": 0.20,
        "module": "profile_morfologico",
        "category": "frente",
        "personality_or_temperament": "temperamento"
    },
    
    "abultamiento_tercio_inferior": {
        "lines": "345-345",
        "text": "ab_t_inf (25% derecho, 21% perfil izq): abultamiento tercio inferior - pensamiento enfocado en actuar o aplicar, analitico",
        "threshold_line": "345",
        "threshold_text": "ab_t_inf (25% derecho, 21% perfil izq)",
        "threshold_value": 0.25,
        "module": "profile_morfologico",
        "category": "frente",
        "personality_or_temperament": "temperamento"
    }
}

# ============================================================================
# MÓDULO: PERFIL ANTROPOMÉTRICO
# NewFeature.md líneas 375-503
# ============================================================================

PROFILE_ANTROPOMETRICO_REFERENCES: Dict[str, Dict[str, Any]] = {
    # Nariz largo (líneas 381-389)
    "nariz_corta": {
        "lines": "385-385",
        "text": "nariz corta: impaciente; solo si diagnóstico de nariz punta hacia arriba tambien esta presente",
        "threshold_line": "379",
        "threshold_text": "ambos perfiles tienen que tener el mismo diagnóstico",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "nariz_largo",
        "personality_or_temperament": "personalidad"
    },
    
    "nariz_protruyente": {
        "lines": "387-387",
        "text": "nariz protruyente: persona inteligente",
        "threshold_line": "379",
        "threshold_text": "ambos perfiles tienen que tener el mismo diagnóstico",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "nariz_largo",
        "personality_or_temperament": "personalidad"
    },
    
    "nariz_normal": {
        "lines": "389-389",
        "text": "nariz normal: no se menciona",
        "threshold_line": "379",
        "threshold_text": "ambos perfiles tienen que tener el mismo diagnóstico",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "nariz_largo",
        "personality_or_temperament": "personalidad"
    },
    
    # Mentón (líneas 405-413)
    "menton_sanguineo": {
        "lines": "409-409",
        "text": "menton sanguineo - persona con impulso a actuar rapidamente",
        "threshold_line": "405",
        "threshold_text": "menton derecho y menton izq",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "menton",
        "personality_or_temperament": "temperamento"
    },
    
    "menton_biloso_linfatico": {
        "lines": "411-411",
        "text": "menton biloso/linfatico - no se menciona",
        "threshold_line": "405",
        "threshold_text": "menton derecho y menton izq",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "menton",
        "personality_or_temperament": "temperamento"
    },
    
    "menton_nervioso": {
        "lines": "413-413",
        "text": "menton nervioso - persona que piensa mucho antes de actuar",
        "threshold_line": "405",
        "threshold_text": "menton derecho y menton izq",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "menton",
        "personality_or_temperament": "temperamento"
    },
    
    # Mandíbula antropométrica (líneas 419-427)
    "mandibula_bilosa_antro": {
        "lines": "423-423",
        "text": "mandibula Bilosa - persona con mucha fuerza para continuar con lo que inicia",
        "threshold_line": "419",
        "threshold_text": "mandibula derecha y mandibula izq",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    "mandibula_sanguinea_antro": {
        "lines": "425-425",
        "text": "mandíbula Sanguínea - persona con mucha fuerza en un inicio o primer impulso pero que le batalla para continuar con dicho impulso.",
        "threshold_line": "419",
        "threshold_text": "mandibula derecha y mandibula izq",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    "mandibula_intermedia": {
        "lines": "427-427",
        "text": "mandibula intermedia sanguineo/bilosa - no se menciona",
        "threshold_line": "419",
        "threshold_text": "mandibula derecha y mandibula izq",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "mandibula",
        "personality_or_temperament": "temperamento"
    },
    
    # Protrusión ocular (líneas 479-489)
    "protusion_positiva": {
        "lines": "485-485",
        "text": "Protusion positiva - persona extrovertida",
        "threshold_line": "479",
        "threshold_text": "ambos diagnosticos tienen que estar presentes y coincidir",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "protusion_ocular",
        "personality_or_temperament": "personalidad"
    },
    
    "protusion_nula": {
        "lines": "487-487",
        "text": "Protusion nula - no se menciona",
        "threshold_line": "479",
        "threshold_text": "ambos diagnosticos tienen que estar presentes y coincidir",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "protusion_ocular",
        "personality_or_temperament": "personalidad"
    },
    
    "protusion_negativa": {
        "lines": "489-489",
        "text": "Protrusion negativa - no se mensiona",
        "threshold_line": "479",
        "threshold_text": "ambos diagnosticos tienen que estar presentes y coincidir",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "protusion_ocular",
        "personality_or_temperament": "personalidad"
    },
    
    # Oreja (líneas 493-501)
    "oreja_normal": {
        "lines": "497-497",
        "text": "oreja normal - no se menciona",
        "threshold_line": "491",
        "threshold_text": "ambos diagnósticos tienen que estar presentes y coincidir",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "oreja",
        "personality_or_temperament": "temperamento"
    },
    
    "oreja_corta": {
        "lines": "499-499",
        "text": "oreja corta - introvertida, abruma fácilmente, funciona mejor absorbiendo pequeñas cantidades de información a la vez",
        "threshold_line": "491",
        "threshold_text": "ambos diagnósticos tienen que estar presentes y coincidir",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "oreja",
        "personality_or_temperament": "temperamento"
    },
    
    "oreja_larga": {
        "lines": "501-501",
        "text": "oreja larga negativa - extrovertida, dispuesta a tomar riesgo, capacidad a absorber mucha información o no abrumarse fácilmente",
        "threshold_line": "491",
        "threshold_text": "ambos diagnósticos tienen que estar presentes y coincidir",
        "threshold_value": None,
        "module": "profile_antropometrico",
        "category": "oreja",
        "personality_or_temperament": "temperamento"
    }
}

# ============================================================================
# TAG ALIASES - Mapeo de tags del modelo a keys de referencias
# ============================================================================
# Los modelos de ML usan tags abreviados (cv, el, al, md, etc.) mientras que
# las referencias usan nombres descriptivos. Este mapeo permite encontrar
# el umbral correcto usando el tag del modelo.
# ============================================================================

MODEL_TAG_TO_REFERENCE_KEY: Dict[str, str] = {
    # Cejas (cj_d, cj_i)
    "cv": "ceja_curva",
    "el": "ceja_inclinada", 
    "rc": "ceja_recta",
    
    # Entrecejo
    "normal": "entrecejo_normal",
    "uniceja": "uniceja",
    "lineas_verticales": "lineas_verticales",
    
    # Párpado (parpado_i, parpado_dr)
    "ptosis": "ptosis",
    "pliegue": "pliegue",
    
    # Ojo (oj_d, oj_i)
    "crl": "ojo_circular",
    "al": "ojo_almendrado",
    "fr": "ojo_fruncido",
    "md": "ojo_media_luna_arriba",
    "md_a": "ojo_media_luna_abajo",
    
    # Oído (o_d, o_i)
    "sp_sl": "oido_tercio_superior",
    "sl": "oido_salido",
    "pg": "oido_pegado",
    "pm": "oido_promedio",
    
    # Nariz - grosor
    "nrml": "nariz_normal",
    "grueso": "nariz_gruesa",
    "delgada": "nariz_delgada",
    
    # Nariz - punta (n)
    "rd": "nariz_punta_redonda",
    "i": "nariz_punta_indiferenciada",
    "pn": "nariz_punta_puntiaguda",
    
    # Pómulo (pml_d, pml_i)
    "salido": "pomulo_salido",
    # "pm": "pomulo_promedio",  # Ya está como oido_promedio, evitar duplicado
    "pl": "pomulo_plano",
    
    # Cachete (cch_d, cch_i)
    "ll": "cachete_lleno",
    "planos": "cachete_plano",
    "hn": "cachete_hundido",
    "lineas_sonriza": "lineas_sonrisa",
    
    # Forma de boca (bc)
    "lunar": "boca_lunar",
    "solar": "boca_solar",
    "mercurial": "boca_mercurial",
    "pursed": "boca_pursed",
    
    # Arco de cupido (ac_d, ac_i)
    "nd": "arco_cupido_no_definido",
    "on": "arco_cupido_marcado",
    "pc": "arco_cupido_triangular",
}

# ============================================================================
# DICCIONARIO MAESTRO - Consolidación de todas las referencias
# ============================================================================

NEWFEATURE_REFERENCES: Dict[str, Dict[str, Any]] = {
    **ESPEJO_ROSTRO_REFERENCES,
    **ESPEJO_FRENTE_REFERENCES,
    **FRONTAL_ANTROPOMETRICO_REFERENCES,
    **FRONTAL_MORFOLOGICO_REFERENCES,
    **PROFILE_MORFOLOGICO_REFERENCES,
    **PROFILE_ANTROPOMETRICO_REFERENCES
}


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_newfeature_reference(diagnosis_name: str, module: str = None) -> Dict[str, Any]:
    """
    Obtiene la referencia de NewFeature.md para un diagnóstico específico.
    
    Args:
        diagnosis_name: Nombre del diagnóstico (puede ser tag del modelo o nombre completo)
        module: Módulo opcional para filtrar búsqueda
        
    Returns:
        Dict con información de referencia, o dict vacío si no se encuentra
    """
    # Buscar exacto
    if diagnosis_name in NEWFEATURE_REFERENCES:
        ref = NEWFEATURE_REFERENCES[diagnosis_name]
        if module is None or ref.get("module") == module:
            return ref
    
    # Buscar usando el mapeo de tags del modelo
    if diagnosis_name in MODEL_TAG_TO_REFERENCE_KEY:
        ref_key = MODEL_TAG_TO_REFERENCE_KEY[diagnosis_name]
        if ref_key in NEWFEATURE_REFERENCES:
            ref = NEWFEATURE_REFERENCES[ref_key]
            if module is None or ref.get("module") == module:
                return ref
    
    # Buscar por variaciones comunes (normalización)
    normalized_name = diagnosis_name.lower().replace("-", "_").replace(" ", "_")
    for key, ref in NEWFEATURE_REFERENCES.items():
        if key.lower().replace("-", "_") == normalized_name:
            if module is None or ref.get("module") == module:
                return ref
    
    # Buscar en FRONTAL_MORFOLOGICO_REFERENCES con tag alias
    tag_lower = normalized_name
    if tag_lower in MODEL_TAG_TO_REFERENCE_KEY:
        mapped_key = MODEL_TAG_TO_REFERENCE_KEY[tag_lower]
        if mapped_key in FRONTAL_MORFOLOGICO_REFERENCES:
            return FRONTAL_MORFOLOGICO_REFERENCES[mapped_key]
    
    # No encontrado - devolver general si existe
    if module == "frontal_morfologico" and "_general" in FRONTAL_MORFOLOGICO_REFERENCES:
        return FRONTAL_MORFOLOGICO_REFERENCES["_general"]
    
    return {
        "lines": "N/A",
        "text": f"Referencia no encontrada para: {diagnosis_name}",
        "threshold_line": "N/A",
        "threshold_text": "N/A",
        "threshold_value": None,
        "module": module or "unknown",
        "category": "unknown",
        "personality_or_temperament": "unknown"
    }


def get_module_references(module_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Obtiene todas las referencias de un módulo específico.
    
    Args:
        module_name: Nombre del módulo (ej: "espejo", "frontal_morfologico")
        
    Returns:
        Dict con todas las referencias del módulo
    """
    return {
        name: ref
        for name, ref in NEWFEATURE_REFERENCES.items()
        if ref.get("module") == module_name
    }


def get_category_references(module_name: str, category_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Obtiene todas las referencias de una categoría específica dentro de un módulo.
    
    Args:
        module_name: Nombre del módulo
        category_name: Nombre de la categoría (ej: "cejas", "rostro")
        
    Returns:
        Dict con todas las referencias de la categoría
    """
    return {
        name: ref
        for name, ref in NEWFEATURE_REFERENCES.items()
        if ref.get("module") == module_name and ref.get("category") == category_name
    }


def list_all_modules() -> List[str]:
    """
    Lista todos los módulos disponibles en las referencias.
    
    Returns:
        Lista de nombres de módulos únicos
    """
    modules = set()
    for ref in NEWFEATURE_REFERENCES.values():
        if "module" in ref and not ref.get("module", "").startswith("_"):
            modules.add(ref["module"])
    return sorted(list(modules))


def list_module_categories(module_name: str) -> List[str]:
    """
    Lista todas las categorías de un módulo.
    
    Args:
        module_name: Nombre del módulo
        
    Returns:
        Lista de nombres de categorías únicas
    """
    categories = set()
    for ref in NEWFEATURE_REFERENCES.values():
        if ref.get("module") == module_name and "category" in ref:
            if not ref.get("category", "").startswith("_"):
                categories.add(ref["category"])
    return sorted(list(categories))

