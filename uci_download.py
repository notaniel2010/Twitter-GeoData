from ucimlrepo import fetch_ucirepo
import pandas as pd
#get data
geo_data = fetch_ucirepo(id = 186)

print(geo_data)