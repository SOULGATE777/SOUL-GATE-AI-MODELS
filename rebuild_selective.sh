#!/bin/bash

# Script para hacer rebuild selectivo de docker-compose
# Uso:
#   ./rebuild_selective.sh frontal        # Solo frontal_prod
#   ./rebuild_selective.sh profile        # Solo profile_prod
#   ./rebuild_selective.sh all            # Ambos
#   ./rebuild_selective.sh frontal/antropometrico  # Servicio específico

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar uso
show_usage() {
    echo -e "${BLUE}Uso:${NC}"
    echo "  $0 frontal              # Rebuild todos los servicios de frontal_prod"
    echo "  $0 profile              # Rebuild todos los servicios de profile_prod"
    echo "  $0 all                  # Rebuild todos los servicios"
    echo "  $0 frontal/antropometrico    # Rebuild servicio específico"
    echo ""
    echo -e "${BLUE}Servicios disponibles:${NC}"
    echo "  frontal_prod: antropometrico, validacion, morfologico, espejo, rotacion, preprocesamiento"
    echo "  profile_prod: validacion, morfologico, preprocesamiento, rotacion, antropometrico"
}

# Función para rebuild de un docker-compose
rebuild_service() {
    local dir=$1
    echo -e "${YELLOW}-----------------------------------${NC}"
    echo -e "${YELLOW}Rebuilding: $dir${NC}"
    echo -e "${YELLOW}-----------------------------------${NC}"

    if [ -f "$dir/docker-compose.yml" ]; then
        cd "$dir"

        # Down de los contenedores existentes
        echo "Stopping containers..."
        docker-compose down

        # Rebuild
        echo "Building images..."
        docker-compose build --no-cache

        # Up de los servicios
        echo "Starting services..."
        docker-compose up -d

        cd - > /dev/null
        echo -e "${GREEN}✓ $dir rebuilt successfully${NC}"
        echo ""
    else
        echo -e "${RED}✗ docker-compose.yml not found in $dir${NC}"
        echo ""
        return 1
    fi
}

# Función para rebuild todos los servicios de una categoría
rebuild_category() {
    local category=$1

    if [ "$category" == "frontal" ]; then
        DIRS=(
            "frontal_prod/antropometrico"
            "frontal_prod/validacion"
            "frontal_prod/morfologico"
            "frontal_prod/espejo"
            "frontal_prod/rotacion"
            "frontal_prod/preprocesamiento"
        )
    elif [ "$category" == "profile" ]; then
        DIRS=(
            "profile_prod/validacion"
            "profile_prod/morfologico"
            "profile_prod/preprocesamiento"
            "profile_prod/rotacion"
            "profile_prod/antropometrico"
        )
    else
        echo -e "${RED}Categoría no válida: $category${NC}"
        return 1
    fi

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Rebuilding ${category}_prod services${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    for dir in "${DIRS[@]}"; do
        rebuild_service "$dir"
    done
}

# Main
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

case "$1" in
    frontal)
        rebuild_category "frontal"
        ;;
    profile)
        rebuild_category "profile"
        ;;
    all)
        rebuild_category "frontal"
        rebuild_category "profile"
        ;;
    frontal_prod/*|profile_prod/*)
        rebuild_service "$1"
        ;;
    *)
        echo -e "${RED}Opción no válida: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac

# Resumen final
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Rebuild Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Checking running containers..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
