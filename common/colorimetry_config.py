# common/colorimetry_config.py
# ============================================================================
# NewFeature.md: Configuración de Rangos RGB (líneas 505-669)
# ============================================================================

from typing import Dict, Tuple, Callable, Optional

# ============================================================================
# NewFeature.md líneas 509-607: 9 Categorías de Color de Ojos
# ============================================================================

EYE_COLOR_RANGES: Dict[str, Dict] = {
    'color_de_ojo_negro/cafe_oscuro': {
        'r_range': (0, 90),      # NewFeature.md línea 510
        'g_range': (0, 80),      # NewFeature.md línea 511
        'b_range': (0, 60),      # NewFeature.md línea 512
        'condition': None
    },
    
    'cafe_claro/hazel': {
        'r_range': (90, 180),    # NewFeature.md línea 516
        'g_range': (70, 140),    # NewFeature.md línea 517
        'b_range': (30, 100),    # NewFeature.md línea 518
        'condition': lambda r, g, b: r > g  # NewFeature.md línea 519
    },
    
    'amarillo': {
        'r_range': (120, 255),   # NewFeature.md línea 523
        'g_range': (120, 255),   # NewFeature.md línea 524
        'b_range': (0, 160),     # NewFeature.md línea 525
        'condition': lambda r, g, b: (
            abs(r - g) / max(r, g, 1) <= 0.30 and  # R y G dentro del 30%
            b < min(r, g) * 0.60                     # B menos del 60%
        )
    },
    
    'verde': {
        'r_range': (0, 160),     # NewFeature.md línea 530
        'g_range': (70, 255),    # NewFeature.md línea 531
        'b_range': (0, 110),     # NewFeature.md línea 532
        'condition': lambda r, g, b: g >= r * 0.80  # G >= R o dentro 20%
    },
    
    'azul_claro/gris': {
        'r_range': (0, 220),     # NewFeature.md línea 535
        'g_range': (80, 255),    # NewFeature.md línea 536
        'b_range': (70, 255),    # NewFeature.md línea 537
        'condition': lambda r, g, b: (
            g >= r and b >= r and  # G y B >= R
            g <= b * 1.40          # G no más del 40% mayor que B
        )
    },
    
    'gris': {
        'r_range': (0, 220),     # NewFeature.md línea 541
        'g_range': (60, 255),    # NewFeature.md línea 542
        'b_range': (70, 255),    # NewFeature.md línea 543
        'condition': lambda r, g, b: g <= b * 1.40  # G no más del 40% mayor que B
    },
    
    'azul_oscuro': {
        'r_range': (0, 80),      # NewFeature.md línea 548
        'g_range': (0, 120),     # NewFeature.md línea 549
        'b_range': (70, 135),    # NewFeature.md línea 550
        'condition': lambda r, g, b: g >= r and b >= g
    },
    
    'azul_intenso/morado': {
        'r_range': (30, 140),    # NewFeature.md línea 557
        'g_range': (0, 135),     # NewFeature.md línea 558
        'b_range': (135, 255),   # NewFeature.md línea 559
        'condition': lambda r, g, b: b > g
    },
    
    'azul_verde': {
        'r_range': (60, 85),     # NewFeature.md línea 563
        'g_range': (70, 170),    # NewFeature.md línea 564
        'b_range': (69, 169),    # NewFeature.md línea 565
        'condition': lambda r, g, b: (
            g > b and 
            r < g * 0.50 and 
            b > r
        )
    }
}

# ============================================================================
# NewFeature.md líneas 613-667: 5 Categorías de Color de Palmas
# ============================================================================

PALM_COLOR_RANGES: Dict[str, Dict] = {
    'rosa/sanguineo-linfatico_oscuro': {
        'r_range': (185, 255),   # NewFeature.md línea 615
        'g_range': (130, 185),   # NewFeature.md línea 616
        'b_range': (130, 185),   # NewFeature.md línea 617
        'condition': lambda r, g, b: r > g * 1.20  # R mayor a G por más de 20%
    },
    
    'rojo/sanguineo': {
        'r_range': (185, 255),   # NewFeature.md línea 621
        'g_range': (0, 145),     # NewFeature.md línea 622
        'b_range': (0, 145),     # NewFeature.md línea 623
        'condition': lambda r, g, b: r > g * 1.20 and r > b * 1.20
    },
    
    'amarillo/nervioso': {
        'r_range': (0, 245),     # NewFeature.md línea 627
        'g_range': (80, 255),    # NewFeature.md línea 628
        'b_range': (0, 160),     # NewFeature.md línea 629
        'condition': lambda r, g, b: (
            abs(r - g) / max(r, g, 1) <= 0.35 and  # R y G dentro del 35%
            b < max(r, g) * 0.80                     # B más del 20% menor
        )
    },
    
    'blanco/linfatico': {
        'r_range': (180, 255),   # NewFeature.md línea 635
        'g_range': (150, 255),   # NewFeature.md línea 636
        'b_range': (105, 255),   # NewFeature.md línea 637
        'condition': lambda r, g, b: sum([r > 150, g > 150, b > 150]) >= 2
    },
    
    'bilioso/cafe_o_oscuro': {
        'r_range': (0, 210),     # NewFeature.md línea 641
        'g_range': (0, 180),     # NewFeature.md línea 642
        'b_range': (0, 255),     # NewFeature.md línea 643
        'condition': lambda r, g, b: sum([r < 175, g < 175, b < 175]) >= 2
    }
}

# ============================================================================
# NewFeature.md: Categorías de Tono de Piel (derivadas de líneas 435-477)
# ============================================================================

SKIN_TONE_CATEGORIES: Dict[str, list] = {
    'rojiza/sanguineo': ['rojo/sanguineo', 'rosa/sanguineo-linfatico_oscuro'],
    'amarillenta/nervioso': ['amarillo/nervioso'],
    'cafe/oscura/bilioso': ['bilioso/cafe_o_oscuro'],
    'palida/blanca/linfatico': ['blanco/linfatico']
}
