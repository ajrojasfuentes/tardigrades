import torch
import pandas as pd
from typing import Dict, Tuple, List, Union, Optional, Any

class Frame:
    """
    Tardigrades Frame: Un tensor etiquetado N-dimensional.
    Combina la potencia de torch.Tensor con la usabilidad de Pandas.
    """
    
    def __init__(self, data: torch.Tensor, dims: Tuple[str, ...], coords: Optional[Dict[str, Any]] = None):
        """
        Args:
            data (torch.Tensor): El tensor de datos subyacente.
            dims (tuple): Nombres de las dimensiones (ej. ('batch', 'time', 'feat')).
            coords (dict, opcional): Diccionario {nombre_dim: [etiquetas]}.
        """
        # Validación básica de dimensiones
        if data.ndim != len(dims):
            raise ValueError(f"Shape mismatch: Tensor tiene {data.ndim} dimensiones, pero se proveyeron {len(dims)} nombres: {dims}")
        
        self._data = data
        self._dims = dims
        self._coords = {}

        # Inicialización de coordenadas
        if coords:
            for dim_name, labels in coords.items():
                if dim_name not in dims:
                    raise ValueError(f"Coordenada '{dim_name}' no existe en las dimensiones {dims}")
                
                # Convertimos todo a pd.Index para búsquedas rápidas (O(1) o O(log n))
                index = pd.Index(labels)
                axis = dims.index(dim_name)
                
                if len(index) != data.shape[axis]:
                    raise ValueError(f"Dimension '{dim_name}' size mismatch: Tensor={data.shape[axis]}, Labels={len(index)}")
                
                self._coords[dim_name] = index

    # --- Propiedades Esenciales ---
    
    @property
    def values(self) -> torch.Tensor:
        """Acceso directo al tensor subyacente (rompe la abstracción, útil para debug)"""
        return self._data

    @property
    def shape(self) -> torch.Size:
        return self._data.shape
    
    @property
    def dims(self) -> Tuple[str, ...]:
        return self._dims
    
    @property
    def device(self) -> torch.device:
        return self._data.device

    def __repr__(self) -> str:
        # Un repr bonito es CRUCIAL para que se sienta como Pandas
        header = f"<tardigrades.Frame {self.dims} {self.shape} on {self.device}>"
        coords_str = "\nCoordinates:\n" + "\n".join([f"  * {k}: {v.values[:5]}... ({len(v)})" for k, v in self._coords.items()])
        return f"{header}{coords_str}\nData:\n{self._data.__repr__()}"

    # --- La Magia de PyTorch (Interoperabilidad) ---
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Permite usar funciones de torch (ej. torch.sin(mi_frame)) devolviendo un nuevo Frame.
        """
        if kwargs is None: kwargs = {}
        
        # 1. Desempaquetar: Obtener los tensores crudos de los argumentos
        # Esto busca cualquier instancia de 'Frame' en los argumentos y saca su ._data
        args_raw = [a._data if hasattr(a, '_data') else a for a in args]
        
        # 2. Ejecutar operación: PyTorch opera sobre los datos crudos
        out_tensor = func(*args_raw, **kwargs)
        
        # 3. Re-empaquetar (La parte difícil):
        # Si el resultado es un Tensor, intentamos reconstruir un Frame.
        # POR AHORA (MVP): Si la forma no cambia, conservamos dims y coords.
        
        # Encontramos el primer Frame en los args para usarlo como referencia de metadatos
        ref_frame = next((a for a in args if isinstance(a, Frame)), None)
        
        if isinstance(out_tensor, torch.Tensor) and ref_frame is not None:
            # Caso simple: Operaciones elemento a elemento (sin, cos, exp, + , -)
            # que no cambian la forma del tensor.
            if out_tensor.shape == ref_frame.shape:
                return Frame(out_tensor, ref_frame.dims, ref_frame._coords)
            
            # Caso reducción: (mean, sum). La forma cambia.
            # Aquí necesitaríamos lógica avanzada para saber qué dimensión desapareció.
            # Para el MVP, devolvemos el tensor crudo si la forma cambia.
            return out_tensor
            
        return out_tensor

    # --- Aritmética Básica (Sobrecarga de operadores) ---
    def __add__(self, other):
        return torch.add(self, other)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __mul__(self, other):
        return torch.mul(self, other)
        
    def __truediv__(self, other):
        return torch.div(self, other)