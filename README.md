# NEC-81 in Python: Numerical Electromagnetics Code (NEC-2)

*Mike Markowski, AB3AP* (`mike.ab3ap@gmail.com`)

* **Documentation & Plot Examples:** https://ab3ap.github.io/nec81/
* **University of Delaware Mirror:** https://udel.edu/~mm/nec81/

A complete Python and NumPy conversion of **NEC-81 (v2.2)** preserving the full Method of Moments (MoM) solver with modern config files, 2D/3D Matplotlib visualization, NSMA `.adf` export, and a PySide6 GUI.

## Features & Installation

* **Core Features:** Full NEC-2 solver core (wires, surface patches, symmetry, transmission lines, loads, ground models, and near-field grids), flexible input formats, interactive PySide6 GUI, and NSMA WG16.99.050 `.adf` export.
* **Quick Install:** 
  ```bash
  pip install "nec81[full] @ git+https://github.com"
  ```
  *(See documentation for release archive installation options).*

## Usage & Configuration

* **Desktop GUI:** Run `nec81` with no arguments or pass a `.nec` / `.cfg` file to launch the six-tab PySide6 interface.
* **Command-Line & Batch Mode:** Use flags like `-o`, `--cli`, `--config FILE`, or `-t` to generate starter configurations and run headlessly. Configuration files (`.cfg`) support plain-text parameters and embedded model cards.

## Full Documentation

For the complete NEC-2 card reference, configuration parameter table, and
sample plots, visit https://ab3ap.github.io/nec81/ or
https://udel.edu/~mm/nec81/.

