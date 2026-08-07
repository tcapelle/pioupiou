#!/usr/bin/env python3

import csv
import time

import requests
import os.path
import re
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

SEPARATOR = ','
JOINER = '|'
QUOTECHAR = '"'
bigbigpouet = ()
test=[]
test_speed=[]
def decode_csv(csv):
    for row in csv:

        current_env = re.match(r"\d\d\d\d-\d\d-\d\d.*", row[0])
        if current_env:
            time = row[0]
            latitude = row[1]
            longitude = row[2]
            wind_speed_min = row[3]
            wind_speed_avg = row[4]
            wind_speed_max = float(row[5])
            wind_heading = float(row[6])
            pressure = row[7]
            # print(wind_speed_max, wind_heading)
            test.append(wind_heading)
            test_speed.append(wind_speed_max)

            plt.polar(wind_heading, wind_speed_max, 'o')
            #plt.text(wind_heading, wind_speed_max, f'({wind_speed_max}, {wind_heading}°)')





# from 2024-01 to 2024-02 format csv
i = 1
year=2014
while year < 2026:
    while i < 13:
        if i==12:
            url = f"http://api.pioupiou.fr/v1/archive/456?start={year}-{i:{'0'}{2}}&stop={year+1}-01&format=csv"
        else:
            url=f"http://api.pioupiou.fr/v1/archive/456?start={year}-{i:{'0'}{2}}&stop={year}-{i+1}&format=csv"

        filepath=f"pioudata/{year}-{i:{'0'}{2}}.csv"

        print(url, " and ", filepath)
        if os.path.exists(filepath):
            # read the file
            # open and read the file after the appending:
            f = open(filepath, "r")
            csvv = csv.reader(f, delimiter=SEPARATOR,quotechar=QUOTECHAR, quoting=csv.QUOTE_ALL)
            decode_csv(csvv)

        else:
            # request and save the file
            rsp = requests.get(url)
            time.sleep(2)
            f = open(filepath, "w")
            f.write(rsp.text)
            f.close()
            csvv = csv.reader(rsp.content.decode('utf-8').splitlines(), delimiter=SEPARATOR, quotechar=QUOTECHAR, quoting=csv.QUOTE_ALL)
            decode_csv(csvv)

        i = i + 1
    year = year + 1

    i=1

# plt.ylim(0,100)
plt.yticks(np. arange(0, 100, step=20))
# plt.show()
fig = px.scatter_polar(r=test_speed, theta=test,direction="counterclockwise" )
fig.show()
