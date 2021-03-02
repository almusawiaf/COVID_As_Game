import math
import matplotlib.pyplot as plt

# Minimum number of vaccines
K = 5

for b in [20, 30, 40, 50]:

    plt.plot([i for i in range(1, 10)], [math.log(i, b) if i >= K else 0 for i in range(1, 10)],
             label = 'Cost per vaccine: $' + str(b))

plt.xlabel('Number of vaccines purchased (in the order of 100K)')
plt.ylabel('Reward')

plt.title('Minimum vaccines needed ' + str(K) + '00,000')
plt.legend()
plt.show()


