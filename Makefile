venv:
	./scripts/rl_training/adopt_uv_source.sh
	uv venv --system-site-packages
	uv sync --group dev
#   ロボットの場合は --group robotで実行する
format: 
	./run_ruff.sh