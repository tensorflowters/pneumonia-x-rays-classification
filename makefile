addPyLib:
	@python3.9 -m pip install ${LIB_NAME}

MODEL_ID ?= 1

train:
	@python3.9 scripts/train_$(MODEL_ID).py

run:
	@python3.9 scripts/run_$(MODEL_ID).py