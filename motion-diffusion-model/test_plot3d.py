import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
frame = motion[0]
for chain in skeleton:
    ax.plot(frame[chain,0], frame[chain,1], frame[chain,2], 'r')
plt.savefig('test_frame.png')
