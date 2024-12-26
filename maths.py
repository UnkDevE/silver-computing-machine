import matplotlib.pyplot as plt

def workingout(l, m):
    xs = []
    for i in range(l):
        xs.append(i + (i % m))
    return xs
        
if __name__ == '__main__':
    plt.plot(workingout(784, 10))
    plt.show()