import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
#import geodata from csv
tweets = pd.read_csv('twitter.csv', nrows=1000000)
#print("Loaded Data")
#print(tweets.head(10))

#plt.scatter(tweets['longitude'], tweets['latitude'], color='red')
#plt.show()


#https://simplemaps.com/data/us-cities
#cities data come from ^^
cities = pd.read_csv('uscities.csv')
#print(cities.head(10))
#filtering list of cities by population - Jan 23 2025 
cities = cities[cities['population'] > 100000]
#print(cities.head(10))

#timing script for performance tracking, start clock after data loaded
start_time = time.time()

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

#count tweets by day of week and visualizing
#tweet_days = tweets['day_of_week'].value_counts()
#ordering the days to start on monday
#tweet_days = tweet_days.sort_index()
#plotting
#tweet_days.plot(kind='bar')
#for i, value in enumerate(tweet_days):
#    plt.text(i, value + 0.1, str(value), ha='center')
#plt.show()


#visualizing tweets by hour
#tweet_hour = tweets['hour'].value_counts()
#ordering hours
#tweet_hour = tweet_hour.sort_index()
#plotting
#tweet_hour.plot(kind='bar')
#for i, value in enumerate(tweet_hour):
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
    ((tweets['hour'] < 9) | (tweets['hour'] > 17)) & (tweets['day_of_week'] < 5), #before 9am local time, after 5pm, and a week day
    1, #week day non working hours
          
            np.where(
            (tweets['hour'] >= 9) & (tweets['hour'] <= 17) & (tweets['day_of_week'] < 5), #after 9am local time, before 5pm, and a week day
            2, #week day working hours
                np.where(
                ((tweets['hour'] < 9) | (tweets['hour'] > 17)) & (tweets['day_of_week'] >= 5), #before 9am local time, after 5pm, and a weekend
                3, #weekend non working hours    
                    np.where(
                    (tweets['hour'] >= 9) & (tweets['hour'] <= 17) & (tweets['day_of_week'] >= 5), #after 9am local time, before 5pm, and a weekend
                    4, #weekend non working hours
                        0 #leave everything else alone
                        
                    
                )    
            )
        )
    )

#plotting tweets by day of week/working hour buckets
# tweet_working_day = tweets['working_hours'].value_counts()
# #ordering the bucket from 0-4
# tweet_working_day = tweet_working_day.sort_index()
# tweet_working_day.plot(kind='bar')
# for i, value in enumerate(tweet_working_day):
#     plt.text(i, value + 0.1, str(value), ha='center')
# plt.show()


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
#defining the Haversine formula which converts degrees to miles
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8 #Radius of the Earth in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

#defining Haversine formual as a vectorized method to convert degrees to miles
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 3958.8 #Radius of the Earth in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
#converting dataframes into numpy arrays
coords_tweets = tweets[['latitude', 'longitude']].to_numpy()
coords_cities = cities[['lat', 'lng']].to_numpy()

#building KDTree using city coordinates
tree = KDTree(coords_cities)

#querying the nearest city for each tweet
distances, indices = tree.query(coords_tweets)

# #initial approach, took 97.78196978569031 seconds with 1,000,000 records
## 99.32264399528503 seconds with vectorized haversine added to top
# #adding index value of city info to tweets
# tweets['nearest_city_index'] = indices
# #adding lat and long info for nearest city
# tweets['nearest_city_name'] = tweets['nearest_city_index'].apply(lambda x: cities.iloc[x]['city'])
# tweets['nearest_city_name'] = tweets['nearest_city_index'].apply(lambda x: cities.iloc[x]['state_name'])
# tweets['nearest_city_latitude'] = tweets['nearest_city_index'].apply(lambda x: cities.iloc[x]['lat'])
# tweets['nearest_city_longitude'] = tweets['nearest_city_index'].apply(lambda x: cities.iloc[x]['lng'])

# #getting ditance from each tweet to nearest city
# tweets['distance_from_tweet_to_city'] = tweets.apply(
#     lambda row: haversine(
#         row['latitude'], row['longitude'],
#         cities.iloc[row['nearest_city_index']]['lat'],
#         cities.iloc[row['nearest_city_index']]['lng']
#     ),
#     axis = 1
# )
# #dropping index column since no longer needed
# tweets.drop(columns=['nearest_city_index'], inplace=True)

##0.819807767868042 seconds for 1,000,000 records with vectorized haversine
#trying a different method to improve performance
#mapping indices to city names
nearest_city_name = cities['city'].iloc[indices].values
nearest_city_state = cities['state_name'].iloc[indices].values
nearest_city_lat = cities['lat'].iloc[indices].values
nearest_city_lon = cities['lng'].iloc[indices].values
#add nearest city name, state, and location to tweet info
tweets['nearest_city_name'] = nearest_city_name
tweets['nearest_city_state'] = nearest_city_state
tweets['nearest_city_lat'] = nearest_city_lat
tweets['nearest_city_lon'] = nearest_city_lon
tweets['distance_from_city_to_tweet'] = haversine_vectorized(
    coords_tweets[:, 0], coords_tweets[:, 1],
    coords_cities[indices, 0], coords_cities[indices, 1]
)

#print(tweets)
#end time of the script, printing total run time
end_time = time.time()
print(f'Runtime: {end_time - start_time} seconds')

metrics = tweets.groupby('hour')['distance_from_city_to_tweet'].mean()
print(metrics)