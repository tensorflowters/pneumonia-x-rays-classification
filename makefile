addPyLib:
	@python3.9 -m pip install ${LIB_NAME}

run:
	@python3.9 scripts/main.py