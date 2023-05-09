activate:
	@conda activate zoidberg2.0

listenv:
	@conda env config vars list

addPyLib:
	@python3.9 -m pip install ${LIB_NAME}