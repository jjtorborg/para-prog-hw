# Plot gen script

import matplotlib.pyplot as p

x = [64, 128, 256, 512, 1024]
y1024 = [0.218,0.201,0.201,0.196,0.203]
y2048 = [0.288,0.251,0.254,0.251,0.248]
y4096 = [0.569,0.446,0.446,0.461,0.474]
y8192 = [1.717,1.250,1.259,1.288,1.345]
y16384 = [6.348,4.436,4.454,4.571,4.817]
y32786 = [24.747,17.188,17.262,17.831,18.803]

p.xlabel("Blocksize (threads per block)")
p.ylabel("Time (seconds)")
p.title("Grid Dimension = 1024")
p.plot(x, y1024)
p.show()

p.xlabel("Blocksize (threads per block)")
p.ylabel("Time (seconds)")
p.title("Grid Dimension = 2048")
p.plot(x, y2048)
p.show()

p.xlabel("Blocksize (threads per block)")
p.ylabel("Time (seconds)")
p.title("Grid Dimension = 4096")
p.plot(x, y4096)
p.show()

p.xlabel("Blocksize (threads per block)")
p.ylabel("Time (seconds)")
p.title("Grid Dimension = 8192")
p.plot(x, y8192)
p.show()

p.xlabel("Blocksize (threads per block)")
p.ylabel("Time (seconds)")
p.title("Grid Dimension = 16384")
p.plot(x, y16384)
p.show()

p.xlabel("Blocksize (threads per block)")
p.ylabel("Time (seconds)")
p.title("Grid Dimension = 32786")
p.plot(x, y32786)
p.show()
