import math
def dual_gradient_energy(x0, x1):
    return sum(pow((x0-x1), 2))