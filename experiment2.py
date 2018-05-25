# coding:utf8
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import general


if __name__ == '__main__':
    fixed_df = pd.read_csv(
        "data/Allstorms.ibtracs_wmo.v03r10.csv",
        delim_whitespace=False)

    # Basin [' SI' ' NA' ' EP' ' SP' ' WP' ' NI' ' SA']
    data = fixed_df[fixed_df['Basin'] == ' SP'].groupby('Season')['Serial_Num'].nunique()

    stormsYears = data.index
    stormsNumbers = data.values

    year0, year1 = stormsYears[0], stormsYears[-1]

    plt.style.use('ggplot')
    plt.scatter(stormsYears, stormsNumbers, s=stormsNumbers)
    plt.xlabel("Рік")
    plt.ylabel("Кількість штормів")
    plt.savefig(general.folderPath2 + "exp2_storms1.png")
    plt.clf()

    plt.plot(stormsYears, stormsNumbers, '-ok')
    plt.xlim(year0, year1)
    plt.xlabel("Рік")
    plt.ylabel("Кількість штормів")
    general.set_grid_to_plot()
    plt.savefig(general.folderPath2 + "exp2_storms2.png")
    plt.clf()

    switchpoint = pm.DiscreteUniform('switchpoint',
                                     lower=0,
                                     upper=len(stormsNumbers) - 1,
                                     doc='Switchpoint[year]')

    avg = np.mean(stormsNumbers)
    early_mean = pm.Exponential('early_mean', beta=1./avg)
    late_mean = pm.Exponential('late_mean', beta=1./avg)

    @ pm.deterministic(plot=False)
    def rate(s=switchpoint, e=early_mean, l=late_mean):
        # Concatenate Poisson means
        out = np.zeros(len(stormsNumbers))
        out[:s] = e
        out[s:] = l
        return out

    storms = pm.Poisson('storms',
                        mu=rate,
                        value=stormsNumbers,
                        observed=True)

    storms_model = pm.Model([storms,
                             early_mean,
                             late_mean, rate])

    strmsM = pm.MCMC(storms_model)
    strmsM.sample(iter=40000, burn=1000, thin=20)

    switchpoint_samples = strmsM.trace('switchpoint')[:]
    early_mean_samples = strmsM.trace('early_mean')[:]
    late_mean_samples = strmsM.trace('late_mean')[:]

    figsize(12.5, 8)
    # histogram of the samples:
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.07)
    n_mths = len(stormsNumbers)
    ax = plt.subplot(311)
    ax.set_autoscaley_on(False)

    plt.hist(early_mean_samples, histtype='stepfilled',
             bins=30, alpha=0.85, label='posterior of $e$',
             color='turquoise', normed=True)

    plt.legend(loc='upper left')
    plt.title(r"""Posterior distributions of the variables $e, l, s$""",
              fontsize=16)

    plt.xlim([2, 12])
    plt.ylim([0, 1.7])
    ax = plt.subplot(312)
    ax.set_autoscaley_on(False)
    plt.hist(late_mean_samples,
             bins=30, alpha=0.85, label="posterior of $l$",
             color="purple", normed=True)

    plt.legend(loc="upper left")
    plt.xlim([2, 12])
    plt.ylim([0, 1.1])
    plt.subplot(313)

    w = 1.0 / switchpoint_samples.shape[0] * np.ones_like(switchpoint_samples)
    plt.hist(switchpoint_samples, bins=range(0, n_mths),
             alpha=1, label=r"posterior of $s$", color="green",
             weights=w, rwidth=2., edgecolor="k")
    plt.xlim([20, n_mths - 20])

    plt.xlabel(r"$s$ (in days)", fontsize=14)
    plt.ylabel("probability")
    plt.legend(loc="upper left")
    plt.savefig(general.folderPath2 + "exp2_posterior_distributions.png")
    plt.clf()

    pm.Matplot.plot(strmsM)
