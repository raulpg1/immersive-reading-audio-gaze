import os
import ast
from dotenv import load_dotenv
load_dotenv(".env")

DATA_DIR = os.getenv("DATA_DIR")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

BLINK_THRESHOLD = float(os.getenv("BLINK_THRESHOLD"))
RIGHT_THRESHOLD = float(os.getenv("RIGHT_THRESHOLD"))
LEFT_THRESHOLD = float(os.getenv("LEFT_THRESHOLD"))

BLACK = ast.literal_eval(os.getenv("BLACK"))
WHITE = ast.literal_eval(os.getenv("WHITE"))
BLUE = ast.literal_eval(os.getenv("BLUE"))
RED = ast.literal_eval(os.getenv("RED"))
GREEN = ast.literal_eval(os.getenv("GREEN"))

RIGHT_EYE_INNER_CORNER = int(os.getenv("RIGHT_EYE_INNER_CORNER"))
RIGHT_EYE_OUTER_CORNER = int(os.getenv("RIGHT_EYE_OUTER_CORNER"))
LEFT_EYE_INNER_CORNER = int(os.getenv("LEFT_EYE_INNER_CORNER"))
LEFT_EYE_OUTER_CORNER = int(os.getenv("LEFT_EYE_OUTER_CORNER"))

RIGHT_EYE_TOP_LID = int(os.getenv("RIGHT_EYE_TOP_LID"))
RIGHT_EYE_BOTTOM_LID = int(os.getenv("RIGHT_EYE_BOTTOM_LID"))
LEFT_EYE_TOP_LID = int(os.getenv("LEFT_EYE_TOP_LID"))
LEFT_EYE_BOTTOM_LID = int(os.getenv("LEFT_EYE_BOTTOM_LID"))

RIGHT_IRIS_POINTS = ast.literal_eval(os.getenv("RIGHT_IRIS_POINTS"))
LEFT_IRIS_POINTS = ast.literal_eval(os.getenv("LEFT_IRIS_POINTS"))

AUDIO_OUTPUT_PATH = os.getenv("AUDIO_OUTPUT_PATH")
AUDIO_LDM_MODEL = os.getenv("AUDIO_LDM_MODEL")
AUDIO_NUM_INFERENCE_STEPS=int(os.getenv("AUDIO_NUM_INFERENCE_STEPS"))
AUDIO_GUIDANCE_SCALE=float(os.getenv("AUDIO_GUIDANCE_SCALE"))