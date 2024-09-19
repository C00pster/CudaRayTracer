NVCC = nvcc
CFLAGS = -O3 --use_fast_math --gpu-architecture=compute_86 --gpu-code=sm_86 -I./src
LDFLAGS = -lassimp

BUILDDIR = build

TARGET = RayTracer
SRCS = src/main.cu

OBJS = $(patsubst src/%.cu, $(BUILDDIR)/%.o, $(SRCS))

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(BUILDDIR)/%.o: src/%.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f $(OBJS) $(TARGET)