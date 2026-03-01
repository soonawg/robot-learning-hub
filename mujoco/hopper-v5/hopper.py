import gymnasium as gym
import time

env = gym.make("Hopper-v5", render_mode="human")
obs, info = env.reset()
print("초기 관찰:", obs)
time.sleep(5)  # 5초 동안 창 유지
env.close()