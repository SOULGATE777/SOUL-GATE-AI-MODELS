#!/usr/bin/env python3
"""
Script de integración automática del ThresholdValidationLogger
en todos los servicios ML afectados por NewFeature.md

Este script modifica los archivos de análisis para integrar el logger de validaciones.
"""

import os
import re
from pathlib import Path

# Servicios a modificar según NewFeature.md
SERVICES_TO_INTEGRATE = {
    "frontal_prod/espejo": {
        "pipeline_file": "app/models/espejo_pipeline.py",
        "main_file": "app/main.py",
        "methods_to_update": [
            "_apply_rostro_menton_decision_tree",
            "_apply_frente_decision_tree"
        ],
        "completed": "partial"  # Frente ya está, falta completar rostro
    },
    "frontal_prod/morfologico": {
        "pipeline_file": "app/models/morfologico_pipeline.py",
        "main_file": "app/main.py",
        "categories": [
            "cejas", "entrecejo", "parpado", "ojo", "oido",
            "nariz_grosor", "punta_nariz", "pomulo", "cachete",
            "forma_boca", "arco_cupido", "tercios_faciales"
        ]
    },
    "frontal_prod/antropometrico": {
        "pipeline_file": "app/models/anthropometric_analyzer.py",
        "main_file": "app/main.py",
        "categories": ["tamaño_ojo", "tamaño_boca", "area_facial"]
    },
    "profile_prod/morfologico": {
        "pipeline_file": "app/models/profile_analysis_pipeline.py",
        "main_file": "app/main.py",
        "categories": [
            "dorso_nariz", "lobulo", "mandibula",
            "submenton", "frente"
        ]
    },
    "profile_prod/antropometrico": {
        "pipeline_file": "app/models/profile_anthropometric_analyzer.py",
        "main_file": "app/main.py",
        "categories": [
            "nariz_largo", "nariz_angulo", "menton",
            "mandibula", "protusion_ocular", "oreja"
        ]
    }
}

def add_imports_to_file(filepath):
    """Agrega los imports necesarios al archivo"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if imports already exist
    if 'ThresholdValidationLogger' in content:
        print(f"  ✓ Imports already present in {filepath}")
        return False
    
    # Find where to add imports (after other common imports)
    import_pattern = r'(from common\.threshold_validator import.*?\n)'
    
    new_imports = """from common.threshold_validator import ThresholdValidator, ModuleType
from common.proportion_analyzer import ProportionAnalyzer
from common.rule_engine import RuleEngine
from common.threshold_validation_logger import ThresholdValidationLogger
from common.newfeature_references import get_newfeature_reference, NEWFEATURE_REFERENCES
"""
    
    if re.search(import_pattern, content):
        content = re.sub(import_pattern, new_imports, content, count=1)
    else:
        # Add after sys.path.insert
        pattern = r'(sys\.path\.insert\(0,.*?\)\n\n)'
        content = re.sub(pattern, r'\1' + new_imports + '\n', content, count=1)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Added imports to {filepath}")
    return True

def add_validation_logger_parameter(filepath, method_name):
    """Agrega el parámetro validation_logger a un método"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to find method definition
    pattern = rf'(def {method_name}\(self,.*?)\):'
    
    def replacement(match):
        params = match.group(1)
        if 'validation_logger' in params:
            return match.group(0)  # Already has parameter
        return params + ', validation_logger=None):'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✓ Added validation_logger parameter to {method_name}")
        return True
    else:
        print(f"  ⚠ Method {method_name} not found or already updated")
        return False

def create_integration_summary():
    """Crea un resumen de la integración"""
    summary = """# Integration Status - Threshold Validation Logging

## Services Status

"""
    
    for service, config in SERVICES_TO_INTEGRATE.items():
        status = config.get('completed', 'pending')
        icon = "✅" if status == "complete" else "🔄" if status == "partial" else "⏳"
        summary += f"### {icon} {service}\n"
        summary += f"- **Status**: {status}\n"
        summary += f"- **Pipeline**: `{config['pipeline_file']}`\n"
        summary += f"- **Main**: `{config['main_file']}`\n"
        
        if 'categories' in config:
            summary += f"- **Categories**: {len(config['categories'])} ({', '.join(config['categories'][:3])}...)\n"
        elif 'methods_to_update' in config:
            summary += f"- **Methods**: {', '.join(config['methods_to_update'])}\n"
        
        summary += "\n"
    
    summary += """
## Next Steps

1. Review modified files
2. Test each service individually
3. Verify validation_log.md is generated
4. Compare logs against NewFeature.md
5. Rebuild Docker containers

## Commands

```bash
# Rebuild specific service
cd /home/mitza/proyectos/SOUL-GATE-AI-MODELS
docker-compose build espejo
docker-compose restart espejo

# Test
curl -X POST -F "file=@test.jpg" http://localhost:8XXX/analyze
# Check: ./frontal_prod/espejo/analysis_logs/YYYY-MM-DD/{uuid}/validation_log.md
```
"""
    
    return summary

def main():
    """Main integration script"""
    print("="*70)
    print("THRESHOLD VALIDATION LOGGER - AUTOMATIC INTEGRATION")
    print("="*70)
    print()
    
    base_path = Path("/home/mitza/proyectos/SOUL-GATE-AI-MODELS")
    
    modified_files = []
    
    for service_name, config in SERVICES_TO_INTEGRATE.items():
        print(f"\n📦 Processing: {service_name}")
        print("-" * 70)
        
        service_path = base_path / service_name
        pipeline_path = service_path / config['pipeline_file']
        
        if not pipeline_path.exists():
            print(f"  ⚠ Pipeline file not found: {pipeline_path}")
            continue
        
        # Add imports
        if add_imports_to_file(str(pipeline_path)):
            modified_files.append(str(pipeline_path))
        
        # Add validation_logger parameters to methods
        if 'methods_to_update' in config:
            for method in config['methods_to_update']:
                if add_validation_logger_parameter(str(pipeline_path), method):
                    modified_files.append(str(pipeline_path))
    
    print("\n" + "="*70)
    print(f"INTEGRATION COMPLETE - {len(set(modified_files))} files modified")
    print("="*70)
    
    # Create summary
    summary = create_integration_summary()
    summary_path = base_path / "docs" / "INTEGRATION_STATUS.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"\n✓ Integration summary saved: {summary_path}")
    print("\nModified files:")
    for f in sorted(set(modified_files)):
        print(f"  - {f}")
    
    print("\n⚠ IMPORTANT: You still need to:")
    print("  1. Add logging calls inside each method")
    print("  2. Update main.py to instantiate and pass validation_logger")
    print("  3. Test each service")
    print("  4. Rebuild Docker containers")

if __name__ == "__main__":
    main()

