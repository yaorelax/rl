import numpy as np
import matplotlib.pyplot as plt

INTERATIONS = 2000
TIMES = 1000
EPSILON = 0.01
EPSILON2 = 0.1
EPSILON3 = 0
mu = [0.2, -0.7, 1.5, 0.5, 1.1, -1.4, -0.1, -1, 0.9, -0.5]

def choose(number):
    return np.random.normal(loc=mu[number], scale=1, size=1)[0]

def main():
    # epsilon == 0.01
    assess = [-5 for _ in range(10)]
    assess_times = [0 for _ in range(10)]
    result = []
    for iter in range(INTERATIONS):
        score = 0
        for time in range(TIMES):
            if np.random.rand() < EPSILON:
                number = np.random.randint(10)
            else:
                number = np.argmax(assess)
            award = choose(number)
            score += award
            assess_times[number] += 1
            assess[number] = assess[number] + (award - assess[number]) / assess_times[number]

        if not iter % 100:
            print('iter:', iter, 'score', score / TIMES)
        result.append(score / TIMES)

    # epsilon == 0.1
    assess = [-5 for _ in range(10)]
    assess_times = [0 for _ in range(10)]
    result2 = []
    for iter in range(INTERATIONS):
        score = 0
        for time in range(TIMES):
            if np.random.rand() < EPSILON2:
                number = np.random.randint(10)
            else:
                number = np.argmax(assess)
            award = choose(number)
            score += award
            assess_times[number] += 1
            assess[number] = assess[number] + (award - assess[number]) / assess_times[number]

        if not iter % 100:
            print('iter2:', iter, 'score', score / TIMES)
        result2.append(score / TIMES)

    # epsilon == 0
    assess = [-5 for _ in range(10)]
    assess_times = [0 for _ in range(10)]
    result3 = []
    for iter in range(INTERATIONS):
        score = 0
        for time in range(TIMES):
            if np.random.rand() < EPSILON3:
                number = np.random.randint(10)
            else:
                number = np.argmax(assess)
            award = choose(number)
            score += award
            assess_times[number] += 1
            assess[number] = assess[number] + (award - assess[number]) / assess_times[number]

        if not iter % 100:
            print('iter3:', iter, 'score', score / TIMES)
        result3.append(score / TIMES)

    #print(assess)
    #print(assess_times)
    plt.ylim(0, 1.75)
    plt.plot(range(0, INTERATIONS, 1), result, color='r', linewidth=0.2)
    plt.plot(range(0, INTERATIONS, 1), result2, color='g', linewidth=0.2)
    plt.plot(range(0, INTERATIONS, 1), result3, color='b', linewidth=0.2)
    plt.legend(['ε=0.01', 'ε=0.1', 'ε=0'])
    plt.show()



if __name__ == '__main__':
    main()
