import numpy as np
import pandas as pd
import pymc
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/Allstorms.ibtracs_wmo.v03r05.csv",
    delim_whitespace=False)

cnt = df[df['Basin'] == ' NA'].groupby('Season') \
    ['Serial_Num'].nunique()
years = cnt.index
y0, y1 = years[0], years[-1]
arr = cnt.values
plt.plot(years, arr, '-ok')
plt.xlim(y0, y1)
plt.xlabel("Year")
plt.ylabel("Number of storms")
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.savefig("storms.png")
plt.clf()

switchpoint = pymc.DiscreteUniform('switchpoint',
                                   lower=0,
                                   upper=len(arr))
early_mean = pymc.Exponential('early_mean', beta=1)
late_mean = pymc.Exponential('late_mean', beta=1)


@pymc.deterministic(plot=False)
def rate(s=switchpoint, e=early_mean, l=late_mean):
    out = np.empty(len(arr))
    out[:s] = e
    out[s:] = l
    return out


storms = pymc.Poisson('storms', mu=rate, value=arr,
                      observed=True)

model = pymc.Model([switchpoint, early_mean,
                    late_mean,
                    rate, storms])

mcmc = pymc.MCMC(model)
mcmc.sample(iter=10000, burn=1000, thin=10)

plt.subplot(311)
plt.plot(mcmc.trace('switchpoint')[:])
plt.ylabel("Switch point")
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.subplot(312)
plt.plot(mcmc.trace('early_mean')[:])
plt.ylabel("Early mean")
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.subplot(313)
plt.plot(mcmc.trace('late_mean')[:])
plt.xlabel("Iteration")
plt.ylabel("Late mean")
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.savefig("markov_chains.png")
plt.clf()

plt.subplot(131)
plt.hist(mcmc.trace('switchpoint')[:] + y0, 15)
plt.xlabel("Switch point")
plt.ylabel("Distribution")
plt.minorticks_on()
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.subplot(132)
plt.hist(mcmc.trace('early_mean')[:], 15)
plt.xlabel("Early mean")
plt.minorticks_on()
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.subplot(133)
plt.hist(mcmc.trace('late_mean')[:], 15)
plt.xlabel("Late mean")
plt.minorticks_on()
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.savefig("distribution.png")
plt.clf()

yp = y0 + mcmc.trace('switchpoint')[:].mean()
em = mcmc.trace('early_mean')[:].mean()
lm = mcmc.trace('late_mean')[:].mean()
print((yp, em, lm))

plt.plot(years, arr, '-ok')
plt.axvline(yp, color='k', ls='--')
plt.plot([y0, yp], [em, em], '-b', lw=3)
plt.plot([yp, y1], [lm, lm], '-r', lw=3)
plt.xlim(y0, y1)
plt.xlabel("Year")
plt.ylabel("Number of storms")
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle='--')
plt.savefig("rate.png")
plt.clf()

graph = pymc.graph.graph(model)
graph.write_png("graph.png")
