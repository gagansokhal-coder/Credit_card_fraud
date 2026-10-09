import logging
import os
from datetime import datetime


# ---------------------------------------------------------
# Create log directory
# ---------------------------------------------------------

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------
# Create log file
# ---------------------------------------------------------

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
)


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d "
           "%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# ---------------------------------------------------------
# Console + File Logger
# ---------------------------------------------------------

console_handler = logging.StreamHandler()

console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[ %(asctime)s ] %(lineno)d "
    "%(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)

logging.getLogger().addHandler(console_handler)