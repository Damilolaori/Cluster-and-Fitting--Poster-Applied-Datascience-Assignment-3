# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 12:48:08 2023

@author: HI
"""
# Importing necessary libraries 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools as iter

# defining functions to read and clean the data
def fetch_scatter_table(filename, filetype, year):
    TABLE = pd.read_csv(filename,skiprows=4)
    TABLE = pd.DataFrame(TABLE[~TABLE[year].isna()][["Country Name","Country Code",year]])
    TABLE.rename(columns = {year: "{}_{}".format(filetype, year)}, inplace = True)
    return TABLE

def fetch_fit_table(filename, country):
    TABLE = pd.read_csv(filename, skiprows=4, header=None)
    TABLE_FIT = pd.DataFrame(TABLE.T)
    TABLE_FIT.columns = TABLE_FIT.iloc[0]
    TABLE_FIT = TABLE_FIT[4:]
    TABLE_FIT.rename(columns = {"Country Name": "Year"}, inplace = True)
    return TABLE_FIT[~TABLE_FIT[country].isna()][["Year",country]]


# Parameters for the indicators for 1990
PPL_CSV = "population.csv"
CO2_CSV = "co2.csv"
GDP_CSV = "gdp.csv"
YEAR = "1990"

# showing the data for population 1990
PPL_TABLE = fetch_scatter_table(PPL_CSV, "PPL", YEAR)
print(PPL_TABLE)

# showing the data for CO2 emission 1990
CO2_TABLE = fetch_scatter_table(CO2_CSV, "CO2", YEAR)
print(CO2_TABLE)

# showing the data for GDP 1990
GDP_TABLE = fetch_scatter_table(GDP_CSV, "GDP", YEAR)
print(GDP_TABLE)

# Merging of the above data and normalisation for year 1990
MAIN_TABLE = PPL_TABLE.merge(CO2_TABLE, on=["Country Name","Country Code"]).merge(GDP_TABLE, on=["Country Name","Country Code"])
MAIN_TABLE["CO2_{}_NORM".format(YEAR)] = MAIN_TABLE["CO2_{}".format(YEAR)]/MAIN_TABLE["PPL_{}".format(YEAR)]
MAIN_TABLE["GDP_{}_NORM".format(YEAR)] = MAIN_TABLE["GDP_{}".format(YEAR)]/MAIN_TABLE["PPL_{}".format(YEAR)]
print(MAIN_TABLE)


# Defining the features to use for clustering
Y = MAIN_TABLE[["GDP_{}_NORM".format(YEAR),"CO2_{}_NORM".format(YEAR)]]

# Running the k-means algorithm to cluster the countries
kmeans = KMeans(n_clusters=2)
kmeans.fit(Y)

# Adding the cluster assignments as a new column to the dataframe
MAIN_TABLE["cluster"] = kmeans.labels_

# Creating a scatter plot to visualize the clusters
plt.figure(figsize = (20,10))
plt.scatter(MAIN_TABLE["GDP_{}_NORM".format(YEAR)], MAIN_TABLE["CO2_{}_NORM".format(YEAR)],  c=MAIN_TABLE["cluster"])

# Showing  the cluster centers on the plot
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.xlabel("GDP per capita (normalized)")
plt.ylabel("CO2 emissions per capita (normalized)")
plt.title("GDP Per Capita VS CO2 Per Capita in {}".format(YEAR))

i=0
while i < len(MAIN_TABLE):
    if (MAIN_TABLE["GDP_{}_NORM".format(YEAR)][i] > 0) & ((MAIN_TABLE["Country Name"][i]=="Kuwait") | (MAIN_TABLE["Country Name"][i]=="Luxembourg")):
        plt.text(MAIN_TABLE["GDP_{}_NORM".format(YEAR)][i], MAIN_TABLE["CO2_{}_NORM".format(YEAR)][i], MAIN_TABLE["Country Name"][i])
    i = i+1

plt.show()

print(metrics.silhouette_score(Y,MAIN_TABLE["cluster"] ))



# Parameters for the indicators for 2019
PPL_CSV = "population.csv"
CO2_CSV = "co2.csv"
GDP_CSV = "gdp.csv"
YEAR = "2019"

# showing the data for population 2019
PPL_TABLE = fetch_scatter_table(PPL_CSV, "PPL", YEAR)
print(PPL_TABLE)

# showing the data for CO2 emission 2019
CO2_TABLE = fetch_scatter_table(CO2_CSV, "CO2", YEAR)
print(CO2_TABLE)

# showing the data for GDP 2019
GDP_TABLE = fetch_scatter_table(GDP_CSV, "GDP", YEAR)
print(GDP_TABLE)

# Merging of the above data and normalisation for year 2019
MAIN_TABLE = PPL_TABLE.merge(CO2_TABLE, on=["Country Name","Country Code"]).merge(GDP_TABLE, on=["Country Name","Country Code"])
MAIN_TABLE["CO2_{}_NORM".format(YEAR)] = MAIN_TABLE["CO2_{}".format(YEAR)]/MAIN_TABLE["PPL_{}".format(YEAR)]
MAIN_TABLE["GDP_{}_NORM".format(YEAR)] = MAIN_TABLE["GDP_{}".format(YEAR)]/MAIN_TABLE["PPL_{}".format(YEAR)]
print(MAIN_TABLE)

# Defining the features to use for clustering
Y = MAIN_TABLE[["GDP_{}_NORM".format(YEAR),"CO2_{}_NORM".format(YEAR)]]

# Running the k-means algorithm to cluster the countries
kmeans = KMeans(n_clusters=3)
kmeans.fit(Y)

# Adding the cluster assignments as a new column to the dataframe
MAIN_TABLE["cluster"] = kmeans.labels_

# Creating a scatter plot to visualize the clusters
plt.figure(figsize = (20,10))
plt.scatter(MAIN_TABLE["GDP_{}_NORM".format(YEAR)], MAIN_TABLE["CO2_{}_NORM".format(YEAR)],  c=MAIN_TABLE["cluster"])

# Showing the cluster centers on the plot
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.xlabel("GDP per capita (normalized)")
plt.ylabel("CO2 emissions per capita (normalized)")
plt.title("GDP Per Capita VS CO2 Per Capita in {}".format(YEAR))

i=0
while i < len(MAIN_TABLE):
    if (MAIN_TABLE["GDP_{}_NORM".format(YEAR)][i] > 0) & ((MAIN_TABLE["Country Name"][i]=="Kuwait") | (MAIN_TABLE["Country Name"][i]=="Luxembourg")):
        plt.text(MAIN_TABLE["GDP_{}_NORM".format(YEAR)][i], MAIN_TABLE["CO2_{}_NORM".format(YEAR)][i], MAIN_TABLE["Country Name"][i])
    i = i+1

plt.show()

print(metrics.silhouette_score(Y,MAIN_TABLE["cluster"] ))

# Importing neccessary libraries
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

# loading the data into a pandas dataframe
COUNTRY = "Kuwait"
CO2_FIT = fetch_fit_table(CO2_CSV, COUNTRY)

# Defining the function to fit: low order polynomials 
def low_poly(x, a, b, c):
    return a + b*x + c*x**2 

# extracting x and y data
A = np.array(CO2_FIT["Year"])
t = A.astype(int)
#t = CO2_FIT["Year"]
B = np.array(CO2_FIT[COUNTRY])
y = B.astype(float)

# using the curve_fit function to fit the exponential growth model
popt, pcov = curve_fit(low_poly, t, y)
# make predictions for the next 20 years
pred_t = np.linspace(t[0], t[-1]+20, 40)
pred_y = low_poly(pred_t, *popt)
sigma = np.sqrt(np.diag(pcov))


# calculating the lower and upper limits of the confidence range using the attached function err_range()
#lower, upper = err_ranges(t, low_poly, popt, sigma)
ci = round(1.96*np.std(y)/np.sqrt(len(t)),2)
lower = np.array(y - ci)
lower = lower.astype(float)
upper = np.array(y + ci)
upper = upper.astype(float)
lower[1]=0

# plotting the best fitting function and the confidence range
plt.figure(figsize = (15,10))
plt.plot(t, y, color='red',label='C02 Emissions')
plt.plot(pred_t, pred_y,color='blue', label='Fitted Line')
plt.fill_between(t, lower, upper, color='yellow', alpha=0.5)
plt.xlabel('Year')
plt.ylabel('CO2 Emissions')
plt.legend()
plt.show()


#defining functions for reading and transposing data
def read_data_excel(excel_url, sheet_name, new_columns, countries):
    data_CO2 = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
    data_CO2 = data_CO2[new_columns]
    data_CO2 = data_CO2.loc[countries]
    data_CO2 = data_CO2.set_index('Country Name', inplace=True)
    
    return data_CO2, data_CO2.transpose()

excel_url = ('https://api.worldbank.org/v2/en/indicator/EN.ATM.CO2E.KT?downloadformat=excel')
sheet_name = 'Data'
data_CO2 = pd.read_excel(excel_url, sheet_name=sheet_name, skiprows=3)
countryFilter = ['Kuwait', 'Luxembourg']
columnFilter = ['Country Name', '1990', '2000', '2010', '2019']
data_CO2 = pd.DataFrame(data_CO2[data_CO2['Country Name'].isin(countryFilter)], columns = columnFilter).transpose()
#data = data.rename(columns=data.iloc[0]).drop(data.index[0])
data_CO2, data_CO2.columns = data_CO2[1:], data_CO2.iloc[0]
print(data_CO2)

# parameters to produce grouped barplots for CO2 Emissions (kt) for Kuwait and
# Luxembourg
data_CO2.plot(kind='bar')
plt.title('CO2 Emissions (kt)')
plt.xlabel('Years')
plt.ylabel('CO2 Emissions')
plt.rcParams["figure.dpi"] = 1000
plt.legend(loc= "upper right")
plt.figure(figsize=(8, 6))
plt.show()j




