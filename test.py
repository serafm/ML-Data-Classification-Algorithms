
from scipy.stats import multinomial


md = multinomial(21, [1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
p = md.pmf([1, 2, 3, 4, 5, 6])
print(p)