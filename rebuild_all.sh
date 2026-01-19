#!/bin/bash

# Script para hacer rebuild de todos los docker-compose en frontal_prod y profile_prod
# Uso: ./rebuild_all.sh

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Docker Compose Rebuild Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Array de directorios con docker-compose
FRONTAL_DIRS=(
    "frontal_prod/antropometrico"
    "frontal_prod/validacion"
    "frontal_prod/morfologico"
    "frontal_prod/espejo"
    "frontal_prod/rotacion"
    "frontal_prod/preprocesamiento"
)

PROFILE_DIRS=(
    "profile_prod/validacion"
    "profile_prod/morfologico"
    "profile_prod/preprocesamiento"
    "profile_prod/rotacion"
    "profile_prod/antropometrico"
)

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
    fi
}

# Rebuild de frontal_prod
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FRONTAL_PROD Services${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

for dir in "${FRONTAL_DIRS[@]}"; do
    rebuild_service "$dir"
done

# Rebuild de profile_prod
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PROFILE_PROD Services${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

for dir in "${PROFILE_DIRS[@]}"; do
    rebuild_service "$dir"
done

# Resumen final
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Rebuild Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Checking running containers..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo -e "${GREEN}All services have been rebuilt and started.${NC}"
