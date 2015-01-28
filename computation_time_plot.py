import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.array(list(map(lambda x: list(map(float, x.split("\t"))),
"""10	100	250	750	2250	6200	10000	12000
0.244	10.666	20.033	45.213	195.22	949.797	1413.038	1836.463
0.001	0.008	0.059	1.201	29.594	595.976	2522.441	8134.59""".split("\n"))))

sns.set(style="white", palette="muted")
b, g, r, p = sns.color_palette("muted", 4)
ax = sns.tsplot(data[1], time=data[0], condition="Learning",color=g)
plt.tight_layout()

ax.fill_between(data[0], 1e-12, data[1], color=g, alpha=0.25)
ax2 = sns.tsplot(data[1] + data[2],condition="Learning + Inversion", time=data[0], color=b)
ax2.fill_between(data[0],  data[1] + data[2], data[1], color=b, alpha=0.25)
plt.legend(loc='upper left')
plt.title("System Computation time vs. Recommendable Documents")
sns.axlabel("# Of Documents", "Time (s)")
plt.savefig("images/computation_time.pdf", bbox_inches="tight")