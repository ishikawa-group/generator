```python
from ase.build import fcc111
import numpy as np

# 表面構造を生成
surf = fcc111('Pt', size=(2, 2, 3), vacuum=10.0)

# 原子の座標を取得
positions = surf.get_positions()
print("Positions:\n", positions)

# 原子番号を取得
atomic_numbers = surf.get_atomic_numbers()
print("Atomic Numbers:\n", atomic_numbers)

# 化学記号を取得
symbols = surf.get_chemical_symbols()
print("Chemical Symbols:\n", symbols)

# セルの情報を取得
cell = surf.get_cell()
print("Cell:\n", cell)

# 表面の情報を取得
info = surf.info
print("Info:\n", info)
```