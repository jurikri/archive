# surface plot
Z = np.array(predcit_save)
X = x
Y = np.arange(Z.shape[0])
X,Y = np.meshgrid(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=1, antialiased=True)
plt.savefig('D://figsave2.png', dpi=1000)
