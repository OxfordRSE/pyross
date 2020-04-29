# Get location of project root
import os, inspect
try:
    frame = inspect.currentframe()
    DIR_PYROSS = os.path.dirname(inspect.getfile(frame))
finally:
    # Always manually delete frame
    # https://docs.python.org/2/library/inspect.html#the-interpreter-stack
    del(frame)


# Import remaining modules
from pyross import (
    contactMatrix,
    control,
    deterministic,
    hybrid,
    inference,
    stochastic,
    forecast,
    utils,
)
