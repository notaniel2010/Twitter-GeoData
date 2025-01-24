import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#import geodata from csv
tweets = pd.read_csv('twitter.csv')#, nrows=100000)
#print("Loaded Data")
#https://simplemaps.com/data/us-cities
#cities data come from ^^
cities = pd.read_csv('uscities.csv')
#print(cities.head(10))
#filtering list of cities by population - Jan 23 2025 
cities = cities[cities['population'] > 100000]

#rows = len(tweets)
#print(rows)
tweets['timestamp'] = tweets['timestamp'].astype(str)
#tweets['timestamp'] = pd.to_datetime(tweets["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

#datetime info is stored in CST, converting all records to be EST
tweets['timestamp_2'] = pd.to_datetime(tweets['timestamp'], format = '%Y%m%d%H%M%S', errors='coerce')
#tweets['timestamp_2'] = tweets['timestamp_2'].dt.strftime('%Y-%d-%m %H:%M:%S')

#types = tweets.dtypes
#print(types)
tweets['standard_time'] = np.where(
    tweets['timezone'] == 1, #1 is Eastern
    tweets['timestamp_2'] + pd.to_timedelta(1, unit = 'h'), #add 1 hour
          
            np.where(
            tweets['timezone'] == 3, #3 is Mountain
            tweets['timestamp_2'] - pd.to_timedelta(1, unit = 'h'), #subtract 1 hour
            
                np.where(
                tweets['timezone'] == 4, #4 is pacific
                tweets['timestamp_2'] - pd.to_timedelta(2, unit = 'h'), #subtract 2 hour
                tweets['timestamp_2']#leave everything else alone
            )
        )
    )
#numeric week day number, 0 is monday, 6 is sunday
tweets['day_of_week'] = tweets['standard_time'].dt.dayofweek
tweets['hour'] = tweets['standard_time'].dt.hour
#count tweets by day of week
#tweet_days = tweets['day_of_week'].value_counts()
#ordering the days to start on monday
#tweet_days = tweet_days.sort_index()
#plotting
#tweet_days.plot(kind='bar')
#for i, value in enumerate(tweet_days):
#    plt.text(i, value + 0.1, str(value), ha='center')

#plt.show()

#24 hr time format
#adding a flag for working hours or not throughout the week, can also be used for week day vs weekend
#1 is week day non working hours, 
#2 is week day working hours
#3 is a weekend non working hours
# 4 is a weekend working hours
#0 is everything else
tweets['working_hours'] = np.where(
    tweets['hour'] < 9 & tweets['hour'] > 17 & tweets['day_of_week'] < 5, #before 9am local time, after 5pm, and a week day
    1, #subtract 1 hour
          
            np.where(
            tweets['hour'] >= 9 & tweets['hour'] <= 17 & tweets['day_of_week'] < 5, #after 9am local time, before 5pm, and a week day
            2, #week day working hours
                np.where(
                tweets['hour'] < 9 & tweets['hour'] > 17 & tweets['day_of_week'] >= 5, #before 9am local time, after 5pm, and a weekend
                3, #weekend non working hours    
                    np.where(
                    tweets['hour'] >= 9 & tweets['hour'] <= 17 & tweets['day_of_week'] >= 5, #after 9am local time, before 5pm, and a weekend
                    4, #weekend non working hours
                        0 #leave everything else alone
                        
                    
                )    
            )
        )
    )


#plt.scatter(tweets['longitude'], tweets['latitude'], color='red')
#plt.show()

#fitting data with k means
#there are 336 cities in the US with populations over 100k in July 1 2023
#kmeans = KMeans(n_clusters = 10, random_state = 0)
#tweets['cluster'] = kmeans.fit_predict(tweets[['latitude', 'longitude']])

#plotting cluster
#plt.scatter(tweets['longitude'], tweets['latitude'], c = tweets['cluster'], cmap = 'viridis', marker = 'o', s = 100)
#plt.show()


#clustering points using dbscan - density based and degrees appart for cluster
#converting to numpy array
#coords = tweets[['latitude', 'longitude']].to_numpy()

#fitting model using euclidian distance, aka degrees
#dbscan = DBSCAN(eps = 2, min_samples = 200, metric = 'haversine')
#tweets['cluster'] = dbscan.fit_predict(np.radians(coords))#converting from lat long to radians for haversine distance to work

#fitting model different method, converting to radians and then measuring in km
#coords_rad = np.radians(coords)
#eps_in_km = 321 #200mi or 321km
#eps_in_radians = eps_in_km / 6371 #earths radius in km
#dbscan = DBSCAN(eps=eps_in_radians, min_samples = 20000, metric = 'haversine')
#tweets['cluster'] = dbscan.fit_predict(coords_rad)

#plotting clusters
#plt.scatter(tweets['longitude'], tweets['latitude'], c=tweets['cluster'], cmap='viridis', marker='o', s=100)
#plt.xlabel('Longitude')
#plt.ylabel('Latitude')
#plt.title('DBSCAN Clustering of Twitter Geospacial Data')
#plt.show()

#print(tweets.head(10))

#distance for each tweet from nearest city, calculate nearest city, then see how the distance changes for the same cities throughout the day
tweets['home_city'] = np.where
