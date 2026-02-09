import pydirectinput
import time
import random

print("Starting in 5 seconds... switch to Cuphead!")
time.sleep(5)

actions = ["left", "right", "down", "s", "shift"]

for i in range(30):
    action = random.choice(actions)
    print("Pressing:", action)

    pydirectinput.press(action)

    time.sleep(0.5)
