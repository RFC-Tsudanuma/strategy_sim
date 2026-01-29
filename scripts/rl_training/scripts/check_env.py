import sys

import rl_env
from stable_baselines3.common.env_checker import check_env

sys.path.append("../..")
import argparse

from main import SimulationMain

parser = argparse.ArgumentParser()
parser.add_argument(
    "--check_env_specification",
    type=bool,
    default=False,
    help="Checking env satisfying the Gymnasium API",
)

import cProfile

# 環境のインスタンスを作成
env = rl_env.StrategySimEnv(action_hz=10)

# 環境が仕様を満たしているかチェック
if parser.parse_args().check_env_specification:
    check_env(env)

max = 100000
cnt_env = SimulationMain(ros2_enable=False, enable_drawing=False, disable_print=True)


def loop():
    i = 0
    while True:
        i += 1
        if i > max:
            break
        _ = cnt_env.step()
        cnt_env.get_observation()


print(f"start for {max} steps...")
import time

start = time.perf_counter()
loop()
elapsed = time.perf_counter() - start
print(f"step {max} times, elapsed time: {elapsed:.2f}")
print(f"step per second: {max / elapsed:.4f}")

print("next profile...")
cProfile.runctx("loop()", globals(), locals(), "Profile.prof")
import pstats

stats = pstats.Stats("Profile.prof")
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats(20)
print("finish profile")
cnt_env.finish_sim()
print("check_env passed!")
