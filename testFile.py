import time

start = time.time()
j = 0
for i in range(100000000):
    j += 1
end = time.time()

T = end-start

print(T)