from playsound import playsound
import os
import time
from pathlib import Path


playsound(str(Path(__file__).parent) + "\\peace.mp3",block=False)
time.sleep(3)