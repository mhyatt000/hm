import matplotlib.pyplot as plt


def scatter(*, data, labels):

    fig, ax = plt.subplots()
    ax.scatter(data[0], data[1])
    ax.set(xlabel=labels[0], ylabel=labels[1])
    plt.show()


def main(): pass


if __name__ == '__main__':
    main()
