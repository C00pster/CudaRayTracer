NVCC = nvcc
CFLAGS_COMMON = -rdc=true --use_fast_math -I./src -O0 -g -G
LDFLAGS = -lassimp

BUILDDIR = build

CFLAGS = $(CFLAGS_COMMON) --gpu-architecture=compute_86 --gpu-code=sm_86
CFLAGS_LAPTOP = $(CFLAGS_COMMON) --gpu-architecture=compute_75 --gpu-code=sm_75

TARGET = RayTracer
SRCS = src/main.cu

OBJS = $(patsubst src/%.cu, $(BUILDDIR)/%.o, $(SRCS))

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: src/%.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

.PHONY: clean laptop

laptop: CFLAGS = $(CFLAGS_LAPTOP)
laptop: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)