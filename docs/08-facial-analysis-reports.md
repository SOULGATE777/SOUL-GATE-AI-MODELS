# 08 — Análisis facial masivo y reportes PDF

**Repositorio:** `/home/mitza/proyectos/Proyecto-Reportes-Masivos`  
**Última actualización:** 2026-05-21

---

## Propósito

**SG Complete Analyzer** — orquestador Python que invoca microservicios ML de **SOUL-GATE-AI-MODELS** directamente (sin API Gateway), agrega respuestas JSON y genera **reportes PDF** unificados para operaciones, investigación y QA.

---

## Script principal

**`unified_analyzer.py`** (recomendado)

```bash
# Frontal + perfil con PDF
python unified_analyzer.py --frontal imagen_frontal.jpg --profile imagen_perfil.jpg

# Solo frontal
python unified_analyzer.py --frontal imagen.jpg --analysis-type frontal

# Sin PDF
python unified_analyzer.py --frontal imagen.jpg --no-pdf

# Ver filtros de datos
python unified_analyzer.py --show-filters
```

---

## Host y puertos

| Configuración | Valor |
|---------------|-------|
| Host ML (hardcoded) | `13.58.240.149` |
| Protocolo | HTTP REST |

| Servicio | Puerto | Endpoints ejemplo |
|----------|--------|-------------------|
| Preproceso frontal | 8014 | `/preprocess-frontal` |
| Morfo frontal | 8000 | `/analyze-face` |
| Antropo frontal | 8001 | `/analyze-anthropometric` |
| Validación frontal | 8002 | `/analyze-validation` |
| Espejo | 8008 | `/analyze-espejo` |
| Rotación frontal | 8012 | `/analyze-frontal-rotation` (collector) |
| Preproceso perfil | 8010 | `/preprocess-profile` |
| Morfo perfil | 8003 | `/analyze-profile-morphological` |
| Antropo perfil | 8004 | `/analyze-profile-anthropometric` |

---

## Flujo `unified_analyzer`

1. Health check de servicios disponibles.
2. **Preprocesamiento prioritario** (8014, 8010).
3. Análisis sobre imagen preprocesada (fallback: original).
4. **`DataFilter`** — elimina claves redundantes (`data_filter.py`).
5. Guardar sesión en `unified_analysis_results/<session_id>/`:
   - `complete_responses.json`
   - metadatos / estadísticas
6. Si no `--no-pdf`: `PDFReportGenerator` → `reporte_unificado_<session_id>.pdf`.

---

## Scripts auxiliares

| Script | Uso |
|--------|-----|
| `response_collector.py` | Colección frontal async (legado) |
| `sg_complete_analyzer.py` | Orquestador frontal legado |
| `profile_complete_analyzer.py` | Solo perfil |
| `pdf_report_generator.py` | PDF desde JSON frontal |
| `endpoint_structure_tester.py` | Prueba estructuras + filtros |
| `test_unified_analyzer.py` | Tests |
| `mass_analyzer.py` | Batch |

---

## Relación con producto

| Aspecto | Producto (web/API) | Reportes masivos |
|---------|-------------------|------------------|
| Entrada | API negocio → Gateway 4051 | Python → ML 800x directo |
| Usuario | Autenticado, planes, DB | Operador / script |
| Salida | UI + `Analysis` Prisma | JSON + PDF en disco |
| Temperamento / narrativa | Gateway + Gemini | **No** incluido por defecto en unified |

---

## Estructura `SG/` (microservicios legacy)

El repo incluye copias bajo `SG/frontal_prod`, `SG/profile_prod`, `SG/body_prod`, `SG/age_prod` para despliegue local con uvicorn (puertos alineados con AI-MODELS).

---

## Referencias

- Pipeline ML: [07-ai-models-pipeline.md](./07-ai-models-pipeline.md)
- `API-GATAWAY/docs/08-reportes-masivos-overview.md`
- `CLAUDE.md` en repo Reportes-Masivos
