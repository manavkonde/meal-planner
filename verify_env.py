import sys

print("Python version:", sys.version)

try:
    import ultralytics
    print("ultralytics version:", ultralytics.__version__)
except ImportError as e:
    print("Failed to import ultralytics:", e)
    sys.exit(1)

try:
    import cv2
    print("opencv-python (cv2) version:", cv2.__version__)
except ImportError as e:
    print("Failed to import cv2:", e)
    sys.exit(1)

try:
    import fastapi
    print("fastapi version:", fastapi.__version__)
except ImportError as e:
    print("Failed to import fastapi:", e)
    sys.exit(1)

try:
    import PIL
    print("pillow (PIL) version:", PIL.__version__)
except ImportError as e:
    print("Failed to import pillow:", e)
    sys.exit(1)

try:
    import numpy as np
    print("numpy version:", np.__version__)
except ImportError as e:
    print("Failed to import numpy:", e)
    sys.exit(1)

try:
    import pandas as pd
    print("pandas version:", pd.__version__)
except ImportError as e:
    print("Failed to import pandas:", e)
    sys.exit(1)

try:
    import sklearn
    print("scikit-learn version:", sklearn.__version__)
except ImportError as e:
    print("Failed to import scikit-learn:", e)
    sys.exit(1)

try:
    import fuzzywuzzy
    print("fuzzywuzzy version successfully imported")
except ImportError as e:
    print("Failed to import fuzzywuzzy:", e)
    sys.exit(1)

print("\nAll core libraries loaded successfully!")
