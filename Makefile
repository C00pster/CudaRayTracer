NVCC = nvcc

CFLAGS_COMMON = --use_fast_math -I./src -O3 -diag-suppress=550
CFLAGS = $(CFLAGS_COMMON) --gpu-architecture=compute_86 --gpu-code=sm_86 -dc
CFLAGS_LAPTOP = $(CFLAGS_COMMON) --gpu-architecture=compute_75 --gpu-code=sm_75

DFLAGS = --gpu-architecture=compute_86 --gpu-code=sm_86

LDFLAGS = -lassimp

BUILDDIR = build

TARGET = RayTracer

SRCS := $(shell find src -type f -name '*.cu')
OBJS = $(patsubst src/%.cu, $(BUILDDIR)/%.o, $(SRCS))
DEVICE_OBJ = $(BUILDDIR)/device.o

all: $(TARGET)

$(TARGET): $(OBJS) $(DEVICE_OBJ)
	$(NVCC) $(CFLAGS_COMMON) $(DFLAGS) $(LDFLAGS) -o $@ $^

$(DEVICE_OBJ): $(OBJS)
	$(NVCC) $(CFLAGS_COMMON) $(DFLAGS) -dlink $^ -o $@

$(BUILDDIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CFLAGS) -c $< -o $@

laptop: CFLAGS = $(CFLAGS_LAPTOP)
laptop: DFLAGS = --gpu-architecture=compute_75 --gpu-code=sm_75
laptop: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean laptop