[len([x for x in sets[i].as_numpy_iterator() if x['label'] != label]) for (i, label) in enumerate(labels)]
[([model.predict(x['image']) for x in sets[i].as_numpy_iterator()], label) for (i, label) in enumerate(labels)]
[ax.plot_wireframe(np.array(i)) for i in range(len(sumtensors)) for t in sumtensors]
x_data = np.arange(0, 1, 0.1)
y_data = np.arange(0, 1, 0.1)
ax.plot_wireframe(np.array(sumtensors))