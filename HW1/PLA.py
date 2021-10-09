import numpy as np

# preprocess
d = np.loadtxt("hw1_train.dat.txt")
d = np.append(np.ones((d.shape[0], 1)), d, axis=1)
x, y = d[:, 0:d.shape[1]-1], d[:, -1]

class PLA:
    def __init__(self, n):
        self.w = np.zeros((n, 1))

    @classmethod
    def sign(cls, d):
        if d <= 0.0:
            return -1.0
        else:
            return 1.0

    def update(self, x, y):
        b = x.shape[0]
        cons_correct_time = 0
        while True:
            sample = np.random.choice(b, 1)
            if abs(self.sign(np.dot(self.w.T, x[sample].T)) - y[sample]) > 1e-6:
                self.update_per_iter(x[sample].T, y[sample])
                cons_correct_time = 0
            else:
                cons_correct_time += 1

            if cons_correct_time >= 5 * b:
                break

    def update_per_iter(self, sample_x, sample_y):
        self.w = self.w + sample_y * sample_x

    def output_w_squared(self):
        return np.dot(self.w.T, self.w)

TIMES=1000

# Question 13.
avg_w_squared = 0.0
for i in range(TIMES):
    model = PLA(x.shape[1])
    model.update(x, y)
    w_squared = model.output_w_squared()
    avg_w_squared += w_squared

avg_w_squared /= TIMES
print("Question 13. ", avg_w_squared)

# Question 14.
altered_x, altered_y = np.array(x), np.array(y)
altered_x *= 2
avg_w_squared = 0.0
for i in range(TIMES):
    model = PLA(x.shape[1])
    model.update(altered_x, altered_y)
    w_squared = model.output_w_squared()
    avg_w_squared += w_squared

avg_w_squared /= TIMES
print("Question 14. ", avg_w_squared)

# Question 15.
altered_x, altered_y = np.array(x), np.array(y)
altered_x /= np.linalg.norm(altered_x, axis=1).reshape((altered_x.shape[0], 1))
avg_w_squared = 0.0
for i in range(TIMES):
    model = PLA(x.shape[1])
    model.update(altered_x, altered_y)
    w_squared = model.output_w_squared()
    avg_w_squared += w_squared

avg_w_squared /= TIMES
print("Question 15. ", avg_w_squared)

# Question 16.
altered_x, altered_y = np.array(x), np.array(y)
altered_x[:, 0] = 0
avg_w_squared = 0.0
for i in range(TIMES):
    model = PLA(x.shape[1])
    model.update(altered_x, altered_y)
    w_squared = model.output_w_squared()
    avg_w_squared += w_squared

avg_w_squared /= TIMES
print("Question 16. ", avg_w_squared)