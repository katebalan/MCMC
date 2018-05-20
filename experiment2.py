import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv(
    "data/Allstorms.ibtracs_wmo.v03r05.csv",
    delim_whitespace=False)

print(file.head())

data = file[file['Basin'] == ' NA'].groupby('Season') \
    ['Serial_Num'].nunique()

print(data.head())
stormsYear = data.index
stormsNumber = data.values

plt.scatter(stormsYear, stormsNumber, s=stormsNumber)
plt.xlabel('Year')
plt.ylabel('Number of Storms')
# plt.show()
plt.savefig("tmp/storms2.png")
plt.clf()
