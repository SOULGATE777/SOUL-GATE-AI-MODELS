"""
Lazy Model Loader - Utilidad para carga bajo demanda de modelos ML

Reduce el consumo de RAM inicial al cargar modelos solo cuando se necesitan,
en lugar de cargarlos todos en el startup de la aplicación.
"""

import time
import gc
import logging
from typing import Optional, Callable, Any, Dict, List
from threading import Lock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LazyModelLoader:
    """
    Cargador lazy (perezoso) de modelos con patrón singleton.

    Características:
    - Carga el modelo solo en la primera solicitud
    - Mantiene referencia en memoria para requests subsecuentes
    - Tracking de último uso para futuras optimizaciones
    - Método de descarga manual para liberar memoria

    Uso:
        loader = LazyModelLoader(load_func=lambda: YourModel())
        model = loader.get_model()
        # ... usar modelo ...
        # Opcionalmente: loader.unload_model()
    """

    def __init__(self, load_func: Callable[[], Any], name: str = "model"):
        """
        Args:
            load_func: Función que carga y retorna el modelo
            name: Nombre descriptivo del modelo (para logging)
        """
        self._load_func = load_func
        self._model: Optional[Any] = None
        self._last_used: Optional[float] = None
        self._lock = Lock()
        self._name = name
        self._load_count = 0

    def get_model(self) -> Any:
        """
        Obtiene el modelo, cargándolo si es necesario.

        Returns:
            El modelo cargado
        """
        with self._lock:
            if self._model is None:
                logger.info(f"🔄 Lazy loading model '{self._name}' (first request)...")
                start_time = time.time()

                self._model = self._load_func()
                self._load_count += 1

                load_time = time.time() - start_time
                logger.info(f"✅ Model '{self._name}' loaded successfully in {load_time:.2f}s")

            # Update last used timestamp
            self._last_used = time.time()

            return self._model

    def unload_model(self) -> bool:
        """
        Descarga el modelo y libera memoria.

        Returns:
            True si se descargó, False si ya estaba descargado
        """
        with self._lock:
            if self._model is not None:
                logger.info(f"🗑️  Unloading model '{self._name}'...")

                # Clear CUDA cache if using PyTorch with CUDA
                if TORCH_AVAILABLE and hasattr(self._model, 'to'):
                    try:
                        if next(self._model.parameters()).is_cuda:
                            torch.cuda.empty_cache()
                            logger.info(f"   Cleared CUDA cache for '{self._name}'")
                    except (StopIteration, AttributeError):
                        pass

                # Delete model reference
                del self._model
                self._model = None
                self._last_used = None

                # Force garbage collection
                gc.collect()

                logger.info(f"✅ Model '{self._name}' unloaded successfully")
                return True

            return False

    def is_loaded(self) -> bool:
        """Verifica si el modelo está cargado en memoria."""
        return self._model is not None

    def get_last_used(self) -> Optional[float]:
        """Retorna el timestamp del último uso."""
        return self._last_used

    def get_load_count(self) -> int:
        """Retorna cuántas veces se ha cargado el modelo."""
        return self._load_count

    def get_stats(self) -> dict:
        """Retorna estadísticas del loader."""
        return {
            "name": self._name,
            "is_loaded": self.is_loaded(),
            "last_used": self._last_used,
            "load_count": self._load_count,
            "seconds_since_last_use": time.time() - self._last_used if self._last_used else None
        }


class MultiModelLoader:
    """
    Gestor para múltiples modelos con lazy loading.

    Útil cuando un servicio tiene varios modelos que pueden cargarse
    de forma independiente.
    """

    def __init__(self):
        self._loaders: Dict[str, LazyModelLoader] = {}
        self._lock = Lock()

    def register_model(self, name: str, load_func: Callable[[], Any]) -> None:
        """
        Registra un modelo para lazy loading.

        Args:
            name: Identificador único del modelo
            load_func: Función que carga el modelo
        """
        with self._lock:
            if name in self._loaders:
                logger.warning(f"Model '{name}' already registered, overwriting...")

            self._loaders[name] = LazyModelLoader(load_func=load_func, name=name)
            logger.info(f"📝 Registered model '{name}' for lazy loading")

    def get_model(self, name: str) -> Any:
        """
        Obtiene un modelo por nombre.

        Args:
            name: Identificador del modelo

        Returns:
            El modelo cargado

        Raises:
            KeyError: Si el modelo no está registrado
        """
        if name not in self._loaders:
            raise KeyError(f"Model '{name}' not registered. Available: {list(self._loaders.keys())}")

        return self._loaders[name].get_model()

    def unload_model(self, name: str) -> bool:
        """Descarga un modelo específico."""
        if name not in self._loaders:
            raise KeyError(f"Model '{name}' not registered")

        return self._loaders[name].unload_model()

    def unload_all(self) -> int:
        """
        Descarga todos los modelos.

        Returns:
            Número de modelos descargados
        """
        count = 0
        for loader in self._loaders.values():
            if loader.unload_model():
                count += 1

        logger.info(f"🗑️  Unloaded {count} models")
        return count

    def get_all_stats(self) -> dict:
        """Retorna estadísticas de todos los modelos."""
        return {
            name: loader.get_stats()
            for name, loader in self._loaders.items()
        }

    def get_loaded_models(self) -> List[str]:
        """Retorna lista de modelos actualmente cargados."""
        return [
            name for name, loader in self._loaders.items()
            if loader.is_loaded()
        ]
