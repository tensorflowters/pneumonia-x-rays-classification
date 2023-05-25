include config/.env
export $(shell sed 's/=.*//' config/.env)

LIB ?=

addPyLib:
	@python3.9 -m pip install ${LIB}

start:
	@python3.9 scripts/main.py

httpServer:
	@python3.9 -m http.server