# Build the pre-compiled BPF object for branch_tracker.
#
# Usage:
#   make              # build branch_tracker.bpf.o
#   make vmlinux.h    # regenerate vmlinux.h from running kernel
#   make clean

CLANG   ?= clang
BPFTOOL ?= bpftool
ARCH    := $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')

CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) -I. -I/usr/include

.PHONY: all clean

all: branch_tracker.bpf.o

branch_tracker.bpf.o: branch_tracker.bpf.c vmlinux.h
	$(CLANG) $(CFLAGS) -c $< -o $@

vmlinux.h:
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

clean:
	rm -f branch_tracker.bpf.o
