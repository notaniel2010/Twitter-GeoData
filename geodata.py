import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import geodata from csv

tweets = pd.read_csv('twitter.csv')#, nrows=1000)
#print("Loaded Data")
#rows = len(tweets)
#print(rows)
tweets['timestamp'] = tweets['timestamp'].astype(str)
#tweets['timestamp'] = pd.to_datetime(tweets["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

#datetime info is stored in CST, converting all records to be EST
tweets['timestamp_2'] = pd.to_datetime(tweets['timestamp'], format = '%Y%m%d%H%M%S', errors='coerce')
#tweets['timestamp_2'] = tweets['timestamp_2'].dt.strftime('%Y-%d-%m %H:%M:%S')

types = tweets.dtypes
print(types)
tweets['standard_time'] = np.where(
    tweets['timezone'] == 1, #1 is Eastern
    tweets['timestamp_2'] + pd.to_timedelta(1, unit = 'h'), #subtract 1 hour
          
            np.where(
            tweets['timezone'] == 3, #3 is Mountain
            tweets['timestamp_2'] + pd.to_timedelta(2, unit = 'h'), #subtract 1 hour
            
                np.where(
                tweets['timezone'] == 4, #4 is Mountain
                tweets['timestamp_2'] + pd.to_timedelta(3, unit = 'h'), #subtract 1 hour
                tweets['timestamp_2']#leave everything else alone
            )
        )
    )

plt.scatter(tweets['longitude'], tweets['latitude'], color='red')
plt.show()

print(tweets.head(10))