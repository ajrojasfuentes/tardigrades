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
        header = f"<tardigrades.Frame {self.dims} {self.shape} on {self.device}>"
        coords_info = []
        for k, v in self._coords.items():
            preview = v.values[:5]
            coords_info.append(f"  * {k} ({len(v)}): {preview}...")
            
        coords_str = "\nCoordinates:\n" + "\n".join(coords_info) if coords_info else ""
        return f"{header}{coords_str}\nData:\n{self._data.__repr__()}"

    # --- Métodos de Selección (La parte tipo Pandas) ---

    def isel(self, **selectors: Union[int, slice, List[int]]) -> Union['Frame', torch.Tensor]:
        """
        Selección por índice entero usando nombres de dimensión.
        Ejemplo: frame.isel(batch=0, time=slice(0, 10))
        """
        # 1. Preparar selectores globales (iniciamos seleccionando TODO)
        idx = [slice(None)] * self._data.ndim
        new_coords = self._coords.copy()
        dims_dropped = []
        
        for dim, selector in selectors.items():
            if dim not in self._dims:
                raise ValueError(f"Dimensión '{dim}' no encontrada en {self._dims}")
            
            axis = self._dims.index(dim)
            idx[axis] = selector
            
            # Gestionar coordenadas resultantes
            if dim in new_coords:
                if isinstance(selector, int):
                    # Si seleccionamos un solo entero, la dimensión desaparece (y su coordenada)
                    del new_coords[dim]
                    dims_dropped.append(dim)
                else:
                    # Si es slice o lista, recortamos la coordenada
                    new_coords[dim] = new_coords[dim][selector]

        # 2. Aplicar la selección al tensor (Slicing Nativo de PyTorch)
        new_data = self._data[tuple(idx)]
        
        # 3. Reconstruir el Frame si quedan dimensiones
        if new_data.ndim > 0:
            # Calcular las nuevas dimensiones (quitando las que se redujeron a escalar)
            new_dims = tuple(d for d in self._dims if d not in dims_dropped)
            
            # Validación de seguridad
            if len(new_dims) != new_data.ndim:
                # Caso borde: a veces slicing complejo cambia dimensiones de forma no obvia
                # Por ahora devolvemos el tensor si hay ambigüedad
                return new_data
                
            return Frame(new_data, new_dims, new_coords)
            
        return new_data

    def sel(self, **selectors: Any) -> Union['Frame', torch.Tensor]:
        """
        Selección por etiqueta (Label-based indexing).
        Ejemplo: frame.sel(ciudad='Madrid', fecha='2023-01-01')
        """
        integer_selectors = {}
        
        for dim, label in selectors.items():
            if dim not in self._coords:
                raise ValueError(f"No hay coordenadas (índice) para la dimensión '{dim}', usa .isel() o agrega coords.")
            
            # Usar Pandas para encontrar la ubicación entera (get_loc)
            try:
                loc = self._coords[dim].get_loc(label)
            except KeyError:
                raise KeyError(f"Etiqueta '{label}' no encontrada en la dimensión '{dim}'")
                
            integer_selectors[dim] = loc
            
        # Delegar el trabajo sucio a .isel
        return self.isel(**integer_selectors)

    # --- La Magia de PyTorch (Interoperabilidad Robusta) ---
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}
        
        # 1. Desempaquetar: Obtener datos crudos, verificando tipos
        args_raw = [a._data if isinstance(a, Frame) else a for a in args]
        
        # 2. Ejecutar operación nativa
        out = func(*args_raw, **kwargs)
        
        # 3. Re-empaquetar
        # Buscamos el primer Frame en los args para usarlo como referencia de metadatos
        ref_frame = next((a for a in args if isinstance(a, Frame)), None)
        
        if isinstance(out, torch.Tensor) and ref_frame is not None:
            # Si la forma no cambia (operaciones element-wise como sin, cos, +, -)
            if out.shape == ref_frame.shape:
                return Frame(out, ref_frame.dims, ref_frame._coords)
            
            # TODO: Aquí se podría implementar lógica para reducciones (sum, mean)
            # detectando qué dimensión desapareció para ajustar dims y coords.
            
        return out

    # --- Aritmética Básica y Reflejada ---
    # Permite: frame + 10  Y TAMBIÉN  10 + frame
    
    def __add__(self, other): return torch.add(self, other)
    def __radd__(self, other): return torch.add(other, self)

    def __sub__(self, other): return torch.sub(self, other)
    def __rsub__(self, other): return torch.sub(other, self)

    def __mul__(self, other): return torch.mul(self, other)
    def __rmul__(self, other): return torch.mul(other, self)
        
    def __truediv__(self, other): return torch.div(self, other)
    def __rtruediv__(self, other): return torch.div(other, self)