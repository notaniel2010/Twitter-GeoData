import pandas as pd
import numpy as np
cities = pd.read_csv('uscities.csv')

#print(cities.head(10))
types = cities.dtypes
print(types)
rows = len(cities)

print(rows)
cities = cities[cities['population'] > 100000]

rows = len(cities)
print(cities)