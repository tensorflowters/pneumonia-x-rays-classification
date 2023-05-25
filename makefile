include config/.env
export $(shell sed 's/=.*//' config/.env)

LIB ?=

addPyLib:
	@python3.9 -m pip install ${LIB}

ifeq ($(TRAINING_MODE),1)
start:
	@python3.9 scripts/main.py
else
start:
	@python3.9 scripts/main.py
endif

startNoLog:
	@python3.9 scripts/main.py

httpServer:
	@python3.9 -m http.server