"""
ACalORAT is an ATcollab module to analytically compute parameters 
related to the orbit response matrix. 

Developed from October 2025 to April 2026 at ALBA Synchrotron by Domènec 
Huerta under the supervision of Zeus Martí. It provides exact analytical 
formulas for optics parameters in the ring and their derivatives.

To install the package, activate your enviroment and do blah blah, download
this folder and run rahr, then you can do as you wish!

To update this documentation, run in your enviroment:

pip install pdoc

and inside the folder of the project run:

pdoc --math ./ACalORAT -o ./docs


"""


from . import AnaORM
from . import Elements
from . import physics
from . import math_utils
from . import numerical
from . import plot_utils
from . import read

__all__ = [
    "AnaORM",
    "Elements",
    "physics",
    "math_utils",
    "numerical",
    "plot_utils",
    "read"
]