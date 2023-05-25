include config/.env
export $(shell sed 's/=.*//' config/.env)

LIB ?=

addPyLib:
	@python3.9 -m pip install ${LIB}

httpServer:
	@python3.9 -m http.server