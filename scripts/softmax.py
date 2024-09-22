def gradient_descent(iterations, learning_rate, init):

    for i in range(iterations):
        d = 2 * init
        init = init - learning_rate * d
    return round(init, 5)

print(gradient_descent(10, 0.01, 2))