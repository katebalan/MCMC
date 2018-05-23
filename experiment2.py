# coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import general


if __name__ == '__main__':
    fixed_df = pd.read_csv(
        "data/Allstorms.ibtracs_wmo.v03r10.csv",
        delim_whitespace=False)

    # print(fixed_df['Basin'].head())

    data = fixed_df[fixed_df['Basin'] == ' SI'].groupby('Season')['Serial_Num'].nunique()

    stormsYears = data.index
    stormsNumbers = data.values

    year0, year1 = stormsYears[0], stormsYears[-1]

    plt.style.use('ggplot')
    plt.scatter(stormsYears, stormsNumbers, s=stormsNumbers)
    plt.xlabel("Рік")
    plt.ylabel("Кількість штормів")
    plt.savefig(general.folderPath + "exp2_storms1.png")
    plt.clf()

    plt.plot(stormsYears, stormsNumbers, '-,k')
    plt.xlim(year0, year1)
    plt.xlabel("Рік")
    plt.ylabel("Кількість штормів")
    general.set_grid_to_plot()
    plt.savefig(general.folderPath + "exp2_storms2.png")
    plt.clf()
