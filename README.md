# NEC-81 (Python / NumPy)

A Python and NumPy implementation of the NEC-81 (NEC-2) antenna modeling code.

## Requirements
* Python >= 3.10
* NumPy >= 1.24
* Optional: Matplotlib >= 3.7 (3D geometry & radiation plots), PySide6 >= 6.5 (GUI)

## Installation from Release Package

From the extracted `nec81/` release folder:

```bash
# Full installation (CLI, plots, and GUI)
pip install "nec81-2.2.0-py3-none-any.whl[full]"

# Core solver only
pip install nec81-2.2.0-py3-none-any.whl
```

*(If installing from source instead of the wheel, run `pip install ".[full]"` from the directory containing `pyproject.toml`.)*

## Usage

### Command Line
```bash
nec81 input.nec output.out
```

### Python API
```python
from nec81.mainsa import runNec81

runNec81('input.nec', 'output.out')
```

