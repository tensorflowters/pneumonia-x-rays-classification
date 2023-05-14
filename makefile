LIB ?= 

addPyLib:
	@python3.9 -m pip install ${LIB}

ID ?= 1

train:
	@python3.9 scripts/x_ray_train_$(ID).py

run:
	@python3.9 scripts/x_ray_run_$(ID).py

httpServer:
	@python3.9 -m http.server