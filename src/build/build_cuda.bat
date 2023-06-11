nvcc ../cuda/raytracing.cu -I ../../include -o cuda_raytracing
del /s /q *.exp
del /s /q *.lib