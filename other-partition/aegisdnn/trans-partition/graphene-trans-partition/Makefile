# Use one of these commands to build the manifest for Bash:
#
# - make
# - make DEBUG=1
# - make SGX=1
# - make SGX=1 DEBUG=1
#
# Use `make clean` to remove Graphene-generated files.

# Listing the programs to be run inside the bash script
PROGRAMS = tee_client

TORCH_LIB ?= /home/ubuntu/libtorch17/libtorch/lib
TORCH_DATA_DIR ?= /home/ubuntu/data/cifar10/
TORCH_DATA ?= '"file:/home/ubuntu/data/cifar10/data_batch_1.bin", "file:/home/ubuntu/data/cifar10/data_batch_2.bin", "file:/home/ubuntu/data/cifar10/data_batch_3.bin", "file:/home/ubuntu/data/cifar10/data_batch_4.bin", "file:/home/ubuntu/data/cifar10/data_batch_5.bin", "file:/home/ubuntu/data/cifar10/test_batch.bin"' 
TORCH_BIN ?= /bin/

# Relative path to Graphene root and key for enclave signing
GRAPHENEDIR ?= /home/ubuntu/soter/graphene/
SGX_SIGNER_KEY ?= $(GRAPHENEDIR)/Pal/src/host/Linux-SGX/signer/enclave-key.pem

ifeq ($(DEBUG),1)
GRAPHENEDEBUG = inline
else
GRAPHENEDEBUG = none
endif

.PHONY: all
all: $(addsuffix .manifest,$(PROGRAMS))
ifeq ($(SGX),1)
all: $(addsuffix .token,$(PROGRAMS))
endif

include $(GRAPHENEDIR)/Scripts/Makefile.configs

# Program dependencies (generate from ldd):
#
# For SGX, the manifest needs to list all the libraries loaded during the
# execution, so that the signer can include the file checksums.
#
# The dependencies are generated from the ldd results.

# We need to replace Glibc dependencies with Graphene-specific Glibc. The Glibc
# binaries are already listed in the manifest template, so we can skip them
# from the ldd results. This list also contains some runtime deps of Bash.
GLIBC_DEPS = linux-vdso /lib64/ld-linux-x86-64 libc libm librt libdl libutil libpthread \
             libselinux libpcre libacl libattr

$(addsuffix .manifest,$(PROGRAMS)): %.manifest: manifest.template
	graphene-manifest \
	-Dlog_level=$(GRAPHENE_LOG_LEVEL) \
	-Darch_libdir=$(ARCH_LIBDIR) \
	-Dentrypoint=$(TORCH_BIN)/$(basename $@) \
	-Dtorch_file=$(TORCH_LIB) \
	-Dtorch_data_dir=$(TORCH_DATA_DIR) \
	-Dtorch_data=$(TORCH_DATA) \
	$< > $@

$(addsuffix .manifest.sgx,$(PROGRAMS)): %.manifest.sgx: %.manifest
	graphene-sgx-sign \
	--key $(SGX_SIGNER_KEY) \
	--output $@ \
	--manifest $<

$(addsuffix .sig,$(PROGRAMS)): %.sig: %.manifest.sgx

$(addsuffix .token,$(PROGRAMS)): %.token: %.sig
	graphene-sgx-get-token \
		--output $(basename $<).token --sig $(basename $<).sig

# Extra executables
pal_loader:
	ln -s $(GRAPHENEDIR)/Runtime/pal_loader $@

.PHONY: regression
regression: all
	@mkdir -p scripts/testdir

	./pal_loader bash.manifest -c "ls" > OUTPUT
	@grep -q "Makefile" OUTPUT && echo "[ Success 1/6 ]"
	@rm OUTPUT

	./pal_loader bash.manifest -c "cd scripts && bash bash_test.sh 1" > OUTPUT
	@grep -q "hello 1" OUTPUT      && echo "[ Success 2/6 ]"
	@grep -q "createdfile" OUTPUT  && echo "[ Success 3/6 ]"
	@grep -q "somefile" OUTPUT     && echo "[ Success 4/6 ]"
	@grep -q "current date" OUTPUT && echo "[ Success 5/6 ]"
	@rm OUTPUT

	./pal_loader bash.manifest -c "cd scripts && bash bash_test.sh 3" > OUTPUT
	@grep -q "hello 3" OUTPUT      && echo "[ Success 6/6 ]"
	@rm OUTPUT

	@rm -rf scripts/testdir


.PHONY: clean
clean:
	$(RM) *.manifest *.manifest.sgx *.token *.sig pal_loader OUTPUT scripts/testdir/*

.PHONY: distclean
distclean: clean
