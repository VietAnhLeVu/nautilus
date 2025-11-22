REGISTRY := local

.DEFAULT_GOAL :=
.PHONY: default
default: out/enclaveos.tar

out:
	mkdir out

out/enclaveos.tar: out \
	$(shell git ls-files \
		src/init \
		src/aws \
        src/hello \
	)
ifndef ENCLAVE_APP
	$(error ENCLAVE_APP is not set. Please provide ENCLAVE_APP variable, e.g., make ENCLAVE_APP=weather-example)
endif
	docker build \
		--tag $(REGISTRY)/enclaveos \
		--progress=plain \
		--platform linux/amd64 \
		--output type=local,rewrite-timestamp=true,dest=out\
		-f Containerfile \
		--build-arg ENCLAVE_APP=$(ENCLAVE_APP) \
		.

# Enclave resource specifications
# Adjust these values based on your application needs
# For intent-classifier with TensorFlow, use at least 2048M memory
ENCLAVE_CPU_COUNT ?= 2
ENCLAVE_MEMORY ?= 2048M

.PHONY: run
run: out/nitro.eif
	sudo nitro-cli \
		run-enclave \
		--cpu-count $(ENCLAVE_CPU_COUNT) \
		--memory $(ENCLAVE_MEMORY) \
		--eif-path out/nitro.eif

.PHONY: run-debug
run-debug: out/nitro.eif
	sudo nitro-cli \
		run-enclave \
		--cpu-count $(ENCLAVE_CPU_COUNT) \
		--memory $(ENCLAVE_MEMORY) \
		--eif-path out/nitro.eif \
		--debug-mode \
		--attach-console

.PHONY: update
update:
	./update.sh

