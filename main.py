import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio
from NBayes import NBayes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import os
import subprocess
from zipfile import ZipFile
import shutil
import sys
import psycopg2

def load_pandas(base_dir): 
    
    files = download_from_kaggle(base_dir)

    temp18 = process_file(files, base_dir)

    cleanup(base_dir)
    print('Complete!')
        
    return temp18

def load_sql(base_dir):
    CONNECTION_STRING = ""
    ## Steps below are the same as in Pandas
    
    files = download_from_kaggle(base_dir)
    process_file(files, base_dir)
    
    ### Set up your SQL conection:
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    print('SQL Connection established')
    SQLCursor = SQLConn.cursor()
    schema_name = 'plane_delays'
    table_name = 'y2018'

    try:
        SQLCursor.execute(f"""DROP TABLE {schema_name}.{table_name};""")
        SQLConn.commit()
    except psycopg2.ProgrammingError:
        print(f"CAUTION: Tablenames not found: {schema_name}.{table_name}")
        SQLConn.rollback()

    SQLCursor = SQLConn.cursor()
    ## Load into DB
    print("Creating Table...")
    SQLCursor.execute(f"""
            CREATE TABLE {schema_name}.{table_name}
            ( FL_DATE date
            , OP_CARRIER varchar(3)
            , OP_CARRIER_FL_NUM int
            , ORIGIN varchar(4)
            , DEST varchar(4)
            , CRS_DEP_TIME timestamp
            , DEP_TIME timestamp
            , DEP_DELAY float
            , TAXI_OUT float
            , TAXI_IN float
            , ARR_DELAY float
            , CANCELLED float
            , CANCELLATION_CODE varchar(2)
            , DIVERTED float
            , CRS_ELAPSED_TIME float
            , ACTUAL_ELAPSED_TIME float
            , AIR_TIME float
            , DISTANCE float
            , CARRIER_DELAY float
            , WEATHER_DELAY float
            , NAS_DELAY float
            , SECURITY_DELAY float
            , LATE_DELAY_TIME float
            , ORIGIN_TZ float
            , ORIGIN_NAME varchar(61)
            , ORIGIN_CITY varchar(27)
            , ORIGIN_LAT float
            , ORIGIN_LONG float
            , DEST_TZ float
            , DEST_NAME varchar(61)
            , DEST_CITY varchar(27)
            , DEST_LAT float
            , DEST_LONG float
            , CRS_ARR_TIME timestamp
            , ARR_TIME timestamp
            , WHEELS_OFF timestamp
            , WHEELS_ON timestamp
            , DELAYED int
            );""")
    SQLConn.commit()

    SQL_STATEMENT = f"""
        COPY {schema_name}.{table_name} FROM STDIN WITH
            CSV 
            HEADER
            DELIMITER AS E'\t';
        """
    print(f"Copying data into {table_name}... this could take a while")
    SQLCursor.copy_expert(sql=SQL_STATEMENT , file=open( os.path.join(base_dir, 'final2018.tdf'), 'r'))
    SQLConn.commit()
    print("Adding permissions...")
    
    SQL_STATEMENT = f"""
    GRANT ALL ON {schema_name}.{table_name} TO STUDENTS;
    """
    SQLCursor.execute(SQL_STATEMENT)
    SQLConn.commit()
    print(f"STUDENTS group has ALL access to {schema_name}.{table_name}")

    cleanup(base_dir)
    print('Complete!')

"""
This function takes our departure
time columns, converts them to
epoch time, and then recalculates
the arrival and 'WHEELS_' columns
convertiing back to datetime
from epoch
"""
def epoch_conversion(df):
    print("Implementing epoch calculations...")
    t = df.copy()

    # convert time intervals to seconds
    hrs_to_seconds = [
        'ORIGIN_TZ','DEST_TZ'
    ]

    for col in hrs_to_seconds:
        t.loc[:, col] = t.loc[:, col].astype('int64')*360
    
    min_to_seconds = [
        'CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME', \
            'TAXI_OUT','TAXI_IN','AIR_TIME'
    ]

    for col in min_to_seconds:
        t.loc[:, col] = t.loc[:, col].astype('int64')*60
        
    # now convert depart times to epoch time
    # with offsets
    t.loc[:, 'DEP_TIME_EPOCH'] = \
        (t.loc[:, 'DEP_TIME'].astype('int64')//1e9) - t.ORIGIN_TZ

    t.loc[:, 'CRS_DEP_TIME_EPOCH'] = \
        (t.loc[:, 'CRS_DEP_TIME'].astype('int64')//1e9) - t.ORIGIN_TZ 

    # now generate epoch times
    # for columns that need it
    t.loc[:, 'CRS_ARR_TIME_EPOCH'] = t.CRS_DEP_TIME_EPOCH + t.CRS_ELAPSED_TIME
    t.loc[:, 'ARR_TIME_EPOCH'] = t.DEP_TIME_EPOCH + t.ACTUAL_ELAPSED_TIME
    t.loc[:, 'WHEELS_OFF_EPOCH'] = t.DEP_TIME_EPOCH + t.TAXI_OUT
    t.loc[:, 'WHEELS_ON_EPOCH'] = t.WHEELS_OFF_EPOCH + t.AIR_TIME

    # now we need to re-adjust for
    # TZ offsets and go back to 
    # standard datetime, local
    # to destination airport
    correct_cols = [
        'CRS_ARR_TIME_EPOCH','ARR_TIME_EPOCH','WHEELS_OFF_EPOCH','WHEELS_ON_EPOCH'
    ]

    for col in correct_cols:
        t.loc[:, col] = pd.to_datetime(t.loc[:, col] + t.DEST_TZ, unit='s')
    
    t = t.drop(['DEP_TIME_EPOCH','CRS_DEP_TIME_EPOCH'], axis=1)


    # now we convert back to original units
    for col in min_to_seconds:
        t.loc[:, col] = t.loc[:, col].astype('int64')/60
    
    for col in hrs_to_seconds:
        t.loc[:, col] = t.loc[:, col].astype('int64')/360
    

    return t

"""
This function converts military time float
to usable 24 hr string 'XX:XX'
"""
def convert_mil_time(x):
    if x == '2400':
        x = '0'*4

    if len(x) == 1:
        x = ('0'*3)+x
    if len(x) == 2:
        x = ('0'*2)+x
    if len(x) == 3:
        x = '0'+x
    
    t = str(x)[0:2]+':'+str(x)[2:4]
    x = t    

    return x

"""
This function converts all the 
military time columns into
usable datetime columns
"""
def convert_time_date_cols(df): 
    t = df.copy()
    print('Converting military times to usable 24 hour format...')
    date_col = 'FL_DATE'
    
    # convert military time
    scheduled_time = ['CRS_ARR_TIME','CRS_DEP_TIME']
    for col in scheduled_time:
        t.loc[:, col] = t.loc[:, col].apply(lambda x: convert_mil_time(str(x)))
    
    actual_time = ['DEP_TIME','ARR_TIME','WHEELS_OFF','WHEELS_ON']
    
    for col in actual_time:
        t.loc[:, col] = t.loc[:, col].apply(lambda x: str(x)[:-2])
        t.loc[:, col] = t.loc[:, col].apply(lambda x: convert_mil_time(x))
    time_cols = actual_time + scheduled_time 

    # combine date and time
    for col in time_cols:
        t.loc[:, col] = t.loc[:, [date_col, col]].agg(' '.join, axis=1)
        t.loc[:, col] = pd.to_datetime(t.loc[:, col], format='%Y-%m-%d %H:%M')
    
    return t

"""
Merge function for kaggle data and airport data
to obtain timezone offsets for each
airport's respective timezone
"""
def timeZones(df, airportdf):
    print("Obtaining timezone offsets...")

    t = df.merge(airportdf, how = 'left', left_on = ['ORIGIN'], right_on = ['IATA'])\
          .merge(airportdf, how = 'left', left_on = ['DEST'], right_on = ['IATA'])
        
    t = t.drop(['IATA_x', 'IATA_y'], axis = 1)
        
    t = t.rename(columns = {
          'Timezone_x': 'ORIGIN_TZ', 
          'Timezone_y':'DEST_TZ',
          'Name_x': 'ORIGIN_NAME',
          'Name_y': 'DEST_NAME',
          'City_x': 'ORIGIN_CITY',
          'City_y': 'DEST_CITY',
          'Latitude_x': 'ORIGIN_LAT',
          'Latitude_y': 'DEST_LAT',
          'Longitude_x': 'ORIGIN_LONG',
          'Longitude_y': 'DEST_LONG'
    })
    
    return t
        
"""
Removes unnamed column
"""
def remove_column(df):
    
    df = df.drop('Unnamed: 27', axis = 1)
    
    return df

"""
This function downloads the data
from kaggle into a temporary
directory
"""
def download_from_kaggle(base_dir):
    data_source = 'yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018'
    tmp_data_dir = os.path.join(base_dir,'tmp_data_dir')
    if os.path.isdir(tmp_data_dir) is False:
        os.mkdir(tmp_data_dir)

    files = []
    try:
        # download files from kaggle
        subprocess.check_call(['kaggle','datasets','download','-p',tmp_data_dir,data_source])

        data = 'airline-delay-and-cancellation-data-2009-2018.zip'
        d_path = os.path.join(tmp_data_dir, data)
        files.append(d_path)
        with ZipFile(d_path, 'r') as z:

            f = '2018.csv'
            path = os.path.join(tmp_data_dir, f)
            files.append(path)
            print(f'Extracting {f}...')
            try:
                z.extract(f, path=tmp_data_dir)
            except:
                print(f'Could not extract {f}')
            
            z.close()
    except subprocess.CalledProcessError:
        print('Make sure you have the kaggle API installed', \
            'and have created an authentication file',
            "from kaggle.com. See 'https://github.com/Kaggle/kaggle-api'",
            'for more information.')
    finally:
        print(f'Extracted {len(files)-1} files from {data}')

    # return path to all downloaded files
    return files

"""
This function removes columns without
necessary time information
"""
def dropNa(df):
    t = df.dropna(axis = 0, subset = ['DEP_TIME','ARR_TIME','WHEELS_OFF',\
                                       'WHEELS_ON','CRS_ARR_TIME','CRS_DEP_TIME',
                                       'ORIGIN_TZ','DEST_TZ','CRS_ELAPSED_TIME','ACTUAL_ELAPSED_TIME', \
            'TAXI_OUT','TAXI_IN','AIR_TIME'])
    
    return t

"""
Helper function to delete unzipped csv files
"""
def cleanup(base_dir):
    p = os.path.join(base_dir, 'tmp_data_dir')
    shutil.rmtree(p, ignore_errors=True)

"""
Function that complies all helper
functions to process a file
"""
def process_file(files, base_dir):

    print('Reading in data...')
    temp18 = pd.read_csv(files[1])
    #saving df without unnamed column
    temp18 = remove_column(temp18)

    # use airport data codes to get the 
    # timezone offsets for dif airports
    data_url = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    
    airports = pd.read_csv(data_url, delimiter=',' , \
                           names=['Airport ID', 'Name', 'City', 'Country','IATA', 'ICAO', 'Latitude', \
                                  'Longitude', 'Altitude','Timezone', 'DST', 'Tz database time zone', \
                                      'Type', 'Source'], header = None)
    airports = airports.loc[:, ['IATA', 'Timezone','Name','City','Latitude','Longitude']]
     
    # aggregate and fix data
    temp18 = timeZones(temp18, airports)
    
    temp18 = dropNa(temp18)
    
    temp18 = convert_time_date_cols(temp18)
    
    temp18 = epoch_conversion(temp18)

    temp18 = temp18.drop(['CRS_ARR_TIME','ARR_TIME','WHEELS_OFF','WHEELS_ON'], axis=1)
    temp18 = temp18.rename(columns= {
        'CRS_ARR_TIME_EPOCH': 'CRS_ARR_TIME',
        'ARR_TIME_EPOCH': 'ARR_TIME',
        'WHEELS_OFF_EPOCH': 'WHEELS_OFF',
        'WHEELS_ON_EPOCH': 'WHEELS_ON'
    })

    temp18 = temp18.assign(DELAYED=0)
    temp18.loc[(temp18.ARR_DELAY > 0), 'DELAYED'] = 1
    
    temp18.to_csv(os.path.join(base_dir, 'final2018.tdf'), sep='\t', index=False)

    return temp18

"""
Tests
"""
def tests(base_dir):
    
    CONNECTION_STRING = ""
    SQLConn = psycopg2.connect(CONNECTION_STRING)
    SQLCursor = SQLConn.cursor()

    temp = base_dir + '/tmp_data_dir'
    
    files = download_from_kaggle(temp)
    process_file(files, temp)
    
    df = pd.read_csv(os.path.join(temp, 'final2018.tdf'), sep='\t')
    # test 1
    SQLCursor.execute("""Select count(*) from plane_delays.y2018;""")
    sql_rows =SQLCursor.fetchall()
    sql_rows = sql_rows[0][0]

    df_rows = df.shape[0]

    assert df_rows == sql_rows

    # test 2
    SQLCursor.execute("""SELECT ORIGIN, count(1) as ct from plane_delays.y2018 group by 1;""")
    ### Make sure that the order is alphabetical (lists of tuples are sorted by first element)
    sql_rows = SQLCursor.fetchall()
    sql_rows = pd.DataFrame(sql_rows, columns=['ORIGIN', 'ct']).sort_values(['ORIGIN'], ascending = True).reset_index(drop=True)

    df_rows = df.ORIGIN.value_counts().to_frame().reset_index().rename(columns={
        'ORIGIN':'ct',
        'index':'ORIGIN'
    }).sort_values(['ORIGIN'], ascending=True).reset_index(drop=True)

    assert df_rows.equals(sql_rows)

""" Plots Code """

"""
Function which calls all plotting 
functions and ensures that proper
libraries are installed
"""
def make_plots(base_dir):
    try:
        # make sure necessary libraries installed and working
        fig = px.scatter_geo()
        path = base_dir + '/fake_fig.png'
        pio.write_image(fig, path, format='png')
        os.remove(path)
    except:
        prompt = '\nThis code requires the python packages:\nplotly\nplotly-orca\npsutil\n' + \
            'which are not base python packages. Would you like to have these ' + \
            'packages installed into your current conda environment?\n\n[y / n]\n\n'
        option = input(prompt)
        if option.lower().strip() == 'y':
            subprocess.check_call(['conda','install','-c','plotly','plotly-orca==1.2.1','psutil','requests'])
        else:
            print('If not, you must quit now and install them into your ',
            'preferred environment. The preferred method of installation is,\n\n'
            'conda install -c plotly plotly-orca==1.2.1 psutil requests\n\n'
            'through a conda virtual environment. Alternate installation instructions ',
            'can be found at\n\nhttps://plotly.com/python/static-image-export/')

    # obtain data from SQL
    CONNECTION_STRING = ""
    SQLconn = psycopg2.connect(CONNECTION_STRING)
    SQLcursor = SQLconn.cursor()
    SQLcursor.execute("""select * from plane_delays.y2018""")
    df = pd.DataFrame( SQLcursor.fetchall() )
    df = convert_cols(df)

    # plot all plots
    plot_delays(df, base_dir)
    top_50_airports = plot_avg_num_flights(base_dir)
    plot_seasonalities(df, base_dir)
    plot_flight_network(df, base_dir, top_50_airports)

"""
Here we will create a plot which displays 
the proportional number of delays over the
course of the busiest day of the year by
flight volume
"""
def plot_delays(df, base_dir):
    print('Constructing delay plots...')
    df_dates = df.groupby(['FL_DATE']).agg({
        'DEP_TIME':'count',
    }).rename(columns={'DEP_TIME':'NUM_FLIGHTS'})

    most_flights_day = \
        (df_dates.loc[(df_dates.NUM_FLIGHTS == df_dates.NUM_FLIGHTS.max()), :]
                 .index)

    dfC = df.set_index('FL_DATE')
    biggest_day = dfC.loc[most_flights_day, :].reset_index()

    biggest_day = biggest_day.assign(hr = biggest_day.DEP_TIME.dt.hour)
   
    delays = (biggest_day.groupby(['ORIGIN','hr'])
                        .agg({
                            'DELAYED': 'sum',
                            'ORIGIN_LAT': 'mean',
                            'ORIGIN_LONG': 'mean' 
                        })
                        .reset_index()
                        .sort_values(['ORIGIN','hr']))

    # when delays are 0, the plot gets confused
    # so we give everyone a head start
    delays.DELAYED = delays.DELAYED + 1
    
    times = [6, 8, 10, 12]
    time_dfs = []
    for time in times:
        time_dfs.append(
            delays.loc[delays.hr == time, :]
        )
    for i in range(len(times)):
        fig = px.scatter_geo(
            time_dfs[i],
            lat='ORIGIN_LAT',
            lon='ORIGIN_LONG',
            scope='usa',
            size='DELAYED',
            hover_name='ORIGIN',
            title=f'Delayed Flights at hour {times[i]}'
            )
        fig.show()
        path = os.path.join(base_dir, f'num_delays_hour_{times[i]}.png')
        pio.write_image(fig, path, format='png') 
        print(f'Plot for hour {times[i]} saved to \n{path}')
    print('Complete!')
        
"""
This plots the average number of
domestic flights per day for
the top 50 airports
"""
def plot_avg_num_flights(base_dir):
    
    print('Constructing flights per day plot...')
    CONNECTION_STRING = ""
    SQLconn = psycopg2.connect(CONNECTION_STRING)
    SQLcursor = SQLconn.cursor()

    # we will select the airports with top 50 num '
    # avg flights per day
    SQLcursor.execute(
        """select avg(num_flights) as avg_num_flights,
            origin, avg(lat) as lat, avg(lon) as lon
        from
            (select count(1) as num_flights, origin,
                avg(origin_lat) as lat,
                avg(origin_long) as lon
            from plane_delays.y2018
            group by fl_date, origin) as iq 
        group by origin
        order by 1;"""
    )

    avg_flights_per_day = SQLcursor.fetchall()

    avg_flights_per_day = pd.DataFrame(avg_flights_per_day)
    # airports with top 50 avg flights per day
    avg_flights_per_day = (avg_flights_per_day.rename(
        columns={
            0: 'avg_flights',
            1: 'ORIGIN',
            2: 'LAT',
            3: 'LONG'
            }
        ).sort_values('avg_flights', ascending=False).head(50))
    
    avg_flights_per_day.avg_flights = \
        avg_flights_per_day.avg_flights.astype('int64')
    fig = px.scatter_geo(
        avg_flights_per_day,
        lat='LAT',
        lon='LONG',
        scope='usa',
        size='avg_flights',
        hover_name='ORIGIN',
        title='Top 50 Average Flights Per Day of US Domestic Air Travel'
    )
    fig.show()
    path = os.path.join(base_dir, 'avg_flights_p_day.png')
    pio.write_image(fig, path, format='png')
    print(f'Plot saved to \n{path}\n\nComplete!')

    return avg_flights_per_day.ORIGIN

"""
Creates various plots
to demonstrate the different 
periods of flight volume
"""
def plot_seasonalities(df, base_dir):
    print('Constructing seasonality plots...')
    months = (df.assign(DEP_MONTH=df.DEP_TIME.dt.month)
                .groupby(['DEP_MONTH'])
                .agg({
                    'DEP_MONTH':'count',
                    'ARR_DELAY': 'mean',
                    'CANCELLED': 'sum',
                    'DIVERTED': 'sum'
                }
    ).rename(columns={'DEP_MONTH':'NUM_FLIGHTS'}))

    weekdays = (df.assign(DEP_DAY=df.DEP_TIME.dt.dayofweek)
                  .groupby(['DEP_DAY'])
                  .agg({
                      'DEP_DAY':'count',
                      'ARR_DELAY': 'mean',
                      'CANCELLED': 'sum',
                      'DIVERTED': 'sum'
                  }
    ).rename(columns={'DEP_DAY':'NUM_FLIGHTS'}))

    df_dates = df.groupby(['FL_DATE']).agg({
        'DEP_TIME':'count',
        'ARR_DELAY': 'mean',
        'CANCELLED': 'sum',
        'DIVERTED': 'sum'
        }
    ).rename(columns={'DEP_TIME':'NUM_FLIGHTS'})

        
    plt.xlabel('Date')
    plt.ylabel('Number of Flights')
    plt.title('Number of Flights Each Day of the Year')
    df_dates['NUM_FLIGHTS'].plot()
    
    path = base_dir + '/tot_num_flights.png'
    plt.savefig(path, format='png')
    print(f'Total number of flights per day saved to \n{path}')
    plt.show()

    plt.xlabel('Months')
    plt.ylabel('Number of Flights')
    plt.title('Number of Flights per Month')
    months['NUM_FLIGHTS'].plot()
    
    path = base_dir + '/flights_per_month.png'
    plt.savefig(path, format='png')
    print(f'Total number of flights per month saved to \n{path}')
    plt.show()
    
    plt.xlabel('Weekdays')
    plt.ylabel('Number of Flights')
    plt.title('Number of Flights per Weekday')
    weekdays['NUM_FLIGHTS'].plot()

    path = base_dir + '/flights_per_weekday.png'
    plt.savefig(path, format='png')
    print(f'Total number of flights per weekday saved to \n{path}')
    plt.show()
    
    
    print('Complete!')

"""
This plots the network structure of
connected airports
"""
def plot_flight_network(df, base_dir, subset):
    dfC = df.set_index('FL_DATE')

    dfJuly = dfC.loc['2018-07', :].reset_index()

    dfRoutes = (dfJuly.groupby(['ORIGIN','DEST'])
                    .agg({'FL_DATE':'count'})
                    .rename(columns={
                        'FL_DATE': 'ct'
                    })
                    .reset_index())

    dfPosO = (dfJuly.groupby('ORIGIN')
                .agg({'ORIGIN_LAT':'mean','ORIGIN_LONG':'mean'})
                .reset_index())
    dfPosD = (dfJuly.groupby('DEST')
                .agg({'DEST_LAT':'mean','DEST_LONG':'mean'})
                .reset_index())

    newRoutes = dfRoutes.merge(dfPosO, on='ORIGIN', how='left').merge(dfPosD, on='DEST',how='left')

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        locationmode = 'USA-states',
        lon = dfPosO['ORIGIN_LONG'],
        lat = dfPosO['ORIGIN_LAT'],
        mode = 'markers',
        marker = dict(
            size = 2,
            color = 'rgb(255, 0, 0)',
            line = dict(
                width = 3,
                color = 'rgba(68, 68, 68, 0)'
            )
        )))

    for i in range(len(newRoutes)):
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'USA-states',
                lon = [newRoutes['ORIGIN_LONG'][i], newRoutes['DEST_LONG'][i]],
                lat = [newRoutes['ORIGIN_LAT'][i], newRoutes['DEST_LAT'][i]],
                mode = 'lines',
                line = dict(
                    width = 1,
                    color = 'rgb(252, 32, 128)'),
                    opacity = float(newRoutes['ct'][i] / newRoutes['ct'].max())
            )
        )

    fig.update_layout(
        title_text = 'July 2018 Flight Paths<br>(Hover for airport names)',
        showlegend = False,
        geo = dict(
            scope = 'north america',
            projection_type = 'azimuthal equal area',
            showland = True,
            landcolor = 'rgb(0, 0, 0)',
            countrycolor = 'rgb(100, 100, 100)',
        ),
    )

    path = os.path.join(base_dir, 'network.png')
    fig.show()
    pio.write_image(fig, path, format='png')

""" Analysis Code """

"""
Function which utilizes a custom
Naive Bayes model to predict whether
or not a flight is delayed. Three
models are created, one for each of
July, February, and April. Also,
only airports in the top 50 of flights
per day are utilized. Model
can be found in NBayes.py
"""
def perform_NBayes_analysis():
    # First need to obtain the top 50 airports
    CONNECTION_STRING = ""
    SQLconn = psycopg2.connect(CONNECTION_STRING)
    SQLcursor = SQLconn.cursor()

    SQLcursor.execute(
        """select avg(num_flights) as avg_num_flights,
            origin, avg(lat) as lat, avg(lon) as lon
        from
            (select count(1) as num_flights, origin,
                avg(origin_lat) as lat,
                avg(origin_long) as lon
            from plane_delays.y2018
            group by fl_date, origin) as iq 
        group by origin
        order by 1;"""
    )

    avg_flights_per_day = SQLcursor.fetchall()
    avg_flights_per_day = pd.DataFrame(avg_flights_per_day)

    avg_flights_per_day = (avg_flights_per_day.rename(
    columns={
        0: 'avg_flights',
        1: 'ORIGIN',
        2: 'LAT',
        3: 'LONG'
    }).sort_values('avg_flights', ascending=False).head(50))

    top_50_airports = avg_flights_per_day.ORIGIN.tolist()

    # We will now obtain the data for each time frame
    july = get_congestion(window=['2018-7-1','2018-7-31']).fillna(0)

    feb = get_congestion(window=['2018-2-1','2018-2-28']).fillna(0)

    april = get_congestion(window=['2018-4-1','2018-4-30']).fillna(0)

    # Focus the scope to our top 50 airports
    july = july.set_index('ORIGIN')
    july = july.loc[top_50_airports, :].reset_index()
    july_y = july.DELAYED

    feb = feb.set_index('ORIGIN')
    feb = feb.loc[top_50_airports, :].reset_index()
    feb_y = feb.DELAYED

    april = april.set_index('ORIGIN')
    april = april.loc[top_50_airports, :].reset_index()
    april_y = april.DELAYED

    multinomial = ['OP_CARRIER','ORIGIN','DEST','HR','DOW']
    gaussian = ['DISTANCE','NUM_FLIGHTS']

    july_x = july.loc[:, multinomial + gaussian]
    feb_x = feb.loc[:, multinomial + gaussian]
    april_x = april.loc[:, multinomial + gaussian]


    jx_train, jx_test, jy_train, jy_test = train_test_split(july_x, july_y) 
    fx_train, fx_test, fy_train, fy_test = train_test_split(feb_x, feb_y) 
    ax_train, ax_test, ay_train, ay_test = train_test_split(april_x, april_y) 

    # Now that we have our 3 dataframes, it is time
    # to train some models!
    models = dict()

    nbj = NBayes()
    nbj.fit(df=jx_train,y=jy_train,multinomial=multinomial,gaussian=gaussian)
    nbj.predict(x_pred=jx_test, y_pred=jy_test)
    models.update({
        'july': nbj
    })

    nbf = NBayes()
    nbf.fit(df=fx_train,y=fy_train,multinomial=multinomial,gaussian=gaussian)
    nbf.predict(x_pred=fx_test, y_pred=fy_test)
    models.update({
        'feb': nbf
    })

    nba = NBayes()
    nba.fit(df=ax_train,y=ay_train,multinomial=multinomial,gaussian=gaussian)
    nba.predict(x_pred=ax_test, y_pred=ay_test)
    models.update({
        'april': nba
    })

    # display performance metrics
    # for each model
    for k, v in models.items():
        print(k, 'model performance:')
        v.get_metrics()

"""
Function for performing 
Logistic Regression analysis
"""
def perform_LogReg_analysis():
    
    july_X, july_Y = prepareDataLogisticRegression(startDate = '2018-7-1', endDate = '2018-7-31')
    february_X,february_Y = prepareDataLogisticRegression(startDate='2018-2-1',endDate='2018-2-28')
    april_X,april_Y = prepareDataLogisticRegression(startDate = '2018-4-1',endDate='2018-4-30')

    july_label_actual, july_predictions = LogReg(july_X,july_Y)

    februrary_label_actual, february_predictions = LogReg(february_X,february_Y)

    april_label_actual, april_predictions = LogReg(april_X,april_Y)

"""
Function for performing
Random Forest analysiss
"""
def perform_RF_analysis():
    # July Predictions and Accuracy
    july_X, july_Y = prepareDataRandomForest(startDate = '2018-7-1', endDate = '2018-7-31')
    july_label_actual, july_predictions = randomForestClassification(july_X,july_Y)
    july_accuracy = calcAccuracy(july_label_actual,july_predictions)
    calcMetrics(july_label_actual,july_predictions)

    # February Predictions and Accuracy
    february_X,february_Y =prepareDataRandomForest(startDate='2018-2-1',endDate='2018-2-28')
    february_label_actual,february_predictions = randomForestClassification(february_X,february_Y)
    february_accuracy = calcAccuracy(february_label_actual,february_predictions)
    calcMetrics(february_label_actual,february_predictions)


    april_X,april_Y=prepareDataRandomForest(startDate = '2018-4-1',endDate='2018-4-30')
    april_label_actual,april_predictions = randomForestClassification(april_X,april_Y)
    april_accuracy = calcAccuracy(april_label_actual,april_predictions)
    calcMetrics(april_label_actual,april_predictions)

"""
Returns a dataframe which only includes
flights from the timw window given, for instance
window=['2018-1-1','2018-1-8]
gives data for first week in January.
It also creates 2 new columns,
HR which represents the hour of day
the flight took off in and 
NUM_FLIGHTS which represents the
number of planes in the originating
airport at time of departure.
"""
def get_congestion(window=None):
    if type(window) is not list: window = [window]
    
    CONNECTION_STRING = ""
    SQLconn = psycopg2.connect(CONNECTION_STRING)
    SQLcursor = SQLconn.cursor()
    # we can modify this to use SQL instead of pandas
    if len(window) > 1:
        query = f"""
        select l.*, r.num_planes 
        from 
        (select *, date_part('hour', dep_time) as hr
        from plane_delays.y2018
        where fl_date between '{window[0]}' and '{window[1]}' ) as l
        left join
        (select origin, fl_date, hr, max(num_planes) as num_planes
        from
            (select origin, fl_date, hr, 
                count(hr) over(
                    partition by origin, fl_date, hr 
                    order by origin, fl_date, hr
                    rows between unbounded preceding and current row
                    ) as num_planes
            from (
                select *,  date_part('hour', dep_time) as hr from plane_delays.y2018
                where fl_date between '{window[0]}' and '{window[1]}') as iq
            where fl_date between '{window[1]}' and '{window[1]}'
            ) as iqq
        group by origin, fl_date, hr
        ) as r
        on l.origin = r.origin and l.fl_date = r.fl_date and l.hr = r.hr
        order by fl_date, hr 
        ;
        """
    else:

        query = f"""
        select origin, fl_date, hr, count(1) as num_planes
        from (
            select *,  date_part('hour',dep_time) as hr from plane_delays.y2018
            where fl_date = '{window[0]}') as iq
        where fl_date = '{window[0]}' 
        group by origin, fl_date, hr
        order by hr
        ;
        """

    SQLcursor.execute(query)
    df = pd.DataFrame(SQLcursor.fetchall())

    df = convert_cols(df)
    df.loc[:, 'DOW'] = df.DEP_TIME.dt.dayofweek

    return df

"""
*This function takes in the features df, and the labels series as input
*With those two parameters, this function will run the data through 
    logistic regression and return a prediction array, as well as various
    statistics explaining the effectiveness of the model
*Note: features = month_X, label = month_Y from prepareData()
    
"""
def LogReg(features,labels):
    print('Logistic Regression Step 1/3: Splitting Data...')
    
    #use sklearn.model_selection.train_test_split to partition the data
    #   into test/training sets
    month_X_train, month_X_test, month_Y_train, month_Y_true = \
        train_test_split(features,labels,train_size = 0.8, test_size = 0.2,\
                         random_state = 1)       
    
    #initiate Logistic Regression Object
    month_logReg = LogisticRegression(random_state = 0, solver = 'saga', dual = False, max_iter = 100)
    
    print('Logistic Regression Step 2/3: Training the Model')
    month_logReg.fit(month_X_train,month_Y_train)
    
    print("Fitting Score: ", month_logReg.score(month_X_train, month_Y_train))
    
    #Make Predictions
    print('Logistic Regressiont Final Step: Making Predictions')
    month_preds = month_logReg.predict(month_X_test)
    
    print("Accuracy Score: ", metrics.accuracy_score(month_Y_true, month_preds))
    
    print("Recall Score: ", metrics.recall_score(month_Y_true, month_preds, average='micro'))
    
    print("Precision Score: ", metrics.precision_score(month_Y_true, month_preds, average='micro'))
    
    print("Balanced Accuracy Score: ", metrics.balanced_accuracy_score(month_Y_true, month_preds))
    
    print("F_1 Score: ", metrics.f1_score(month_Y_true, month_preds, average = 'micro'))
    
    return month_Y_true, month_preds

"""
*This function takes in the features df, and the labels series as input
*With those two parameters, this function will run the data through 
    a random forest and return a prediction array, as well as various
    statistics explaining the effectiveness of the model
*Note: features = month_X, label = month_Y from prepareData()
* This function will return the Y_test data, and the predictions to be compared
    or thrown into another function that calculates the accruacy of the 
    predictions
    
"""
def randomForestClassification(features,labels):
    print('RandForestRegression Step 1/4: Splitting Data...')
    
    #use sklearn.model_selection.train_test_split to partition the data
    #   into test/training sets
    #Note: month_Y_true = Y_test
    month_X_training, month_X_test, month_Y_training, month_Y_true = \
        train_test_split(features,labels,train_size = 0.8, test_size = 0.2,\
                         random_state = 1)
    
    #initiate RandomForestRegressor Object
    month_rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
    
    #Train the Model
    #   have to pass an array or a sparse matric to fit
    print('Random Forest Step 2/4: Converting to numPy Arrays')
    mnth_X_training = np.array(month_X_training)
    mnth_X_test = np.array(month_X_test)
    
    print('Random Forest Step 3/4: Training the Model')
    month_rf.fit(mnth_X_training,month_Y_training)
    print("Fitting Score:", month_rf.score(mnth_X_training,month_Y_training))
    
    #Make Predictions
    print('Random Forest Final Step: Making Predictions')
    month_preds = month_rf.predict(mnth_X_test)
    print('Predict Score:', month_rf.score(mnth_X_test, month_Y_true))
    
    return month_Y_true, month_preds


"""
Function which names all columns
of the tuples returned by SQL
"""
def convert_cols(df):
    
    cols = ['FL_DATE'
            , 'OP_CARRIER'
            , 'OP_CARRIER_FL_NUM'
            , 'ORIGIN' 
            , 'DEST' 
            , 'CRS_DEP_TIME' 
            , 'DEP_TIME' 
            , 'DEP_DELAY' 
            , 'TAXI_OUT'
            , 'TAXI_IN'
            , 'ARR_DELAY'
            , 'CANCELLED' 
            , 'CANCELLATION_CODE'
            , 'DIVERTED' 
            , 'CRS_ELAPSED_TIME'
            , 'ACTUAL_ELAPSED_TIME'
            , 'AIR_TIME'
            , 'DISTANCE'
            , 'CARRIER_DELAY'
            , 'WEATHER_DELAY'
            , 'NAS_DELAY'
            , 'SECURITY_DELAY'
            , 'LATE_DELAY_TIME'
            , 'ORIGIN_TZ' 
            , 'ORIGIN_NAME'
            , 'ORIGIN_CITY'
            , 'ORIGIN_LAT'
            , 'ORIGIN_LONG'
            , 'DEST_TZ'
            , 'DEST_NAME'
            , 'DEST_CITY'
            , 'DEST_LAT' 
            , 'DEST_LONG'
            , 'CRS_ARR_TIME'
            , 'ARR_TIME' 
            , 'WHEELS_OFF'
            , 'WHEELS_ON' 
            , 'DELAYED'
            , 'HR'
            , 'NUM_FLIGHTS'
            ]

    col_map = dict()
    for i in range(len(cols)):
        col_map[i] = cols[i]

    df.rename(columns=col_map, inplace=True)

    return df

'''
This Function takes in a start and end date as parameters. This range of 
dates will pull all needed data for the given time frame.
This Function will return month_X (dataFrame) and month_Y (Series)
The returned values will be used to create train and test data
**Note: when calling function -> X,Y = prepareData('2018-7-1','2018-7-31')
'''
def prepareDataLogisticRegression(startDate, endDate):
    print('Preparing Data for dates in range',startDate,':',endDate)
    #calculate top50 airports for later use, save to series
    print('Preparing Data Step 1/4: Populating 50 busiest airports')
    avg_flights_per_day = top_50_airports()
    topFiftyAirports = avg_flights_per_day.ORIGIN
    
    #get dataFrame filled with data from the time period of the parameters
    month_delays = get_congestion(window = [startDate,endDate]).fillna(0)
    month_delays = month_delays[month_delays['ORIGIN'].isin(topFiftyAirports)]
    month_label = month_delays.DELAYED
    print('Preparing Data Step 2/4: Dropping Columns')
    month_delays = month_delays.drop(columns = ['FL_DATE','DELAYED'])
    
    # cannot fit model with datetime type columns
    # create list of datetime columns
    print('Preparing Data Step 3/4: Fixing Column Types')
    dtimeCols = ['CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','WHEELS_OFF','WHEELS_ON']
    for col in dtimeCols:
        month_delays.loc[:,col] = month_delays[col].dt.strftime('%Y-%m-%d')

    cols = ['OP_CARRIER','ORIGIN','DEST','HR','DOW','DISTANCE','NUM_FLIGHTS']
    month_delays = month_delays.loc[:, cols]

    print('Preparing Data Step Final Step: One-Hot Encoding')
    month_delays = pd.get_dummies(month_delays)

    return month_delays, month_label

'''
This Function takes in a start and end date as parameters. This range of 
dates will pull all needed data for the given time frame.
This Function will return month_X (dataFrame) and month_Y (Series)
The returned values will be used to create train and test data
Note: when calling function -> X,Y = prepareData('2018-7-1','2018-7-31')
for Random Forest
'''
def prepareDataRandomForest(startDate, endDate):
    print('Preparing Data for dates in range',startDate,':',endDate)
    #calculate top50 airports for later use, save to series
    print('Preparing Data Step 1/4: Populating 50 busiest airports')
    avg_flights_per_day = top_50_airports()
    topFiftyAirports = avg_flights_per_day.ORIGIN
    
    #get dataFrame filled with data from the time period of the parameters
    month_delays = get_congestion(window = [startDate,endDate]).fillna(0)
    month_delays = month_delays[month_delays['ORIGIN'].isin(topFiftyAirports)]
    month_label = month_delays.DELAYED
    print('Preparing Data Step 2/4: Dropping Columns')
    month_delays = month_delays.drop(columns = ['FL_DATE','DELAYED'])
    
    # #cannot fit model with datetime type columns
    #     #create list of fatetime columns
    print('Preparing Data Step 3/4: Fixing Column Types')
    dtimeCols = ['CRS_DEP_TIME','DEP_TIME','CRS_ARR_TIME','ARR_TIME','WHEELS_OFF','WHEELS_ON']
    for col in dtimeCols:
        month_delays.loc[:,col] = month_delays[col].dt.strftime('%Y-%m-%d')
    
    print('Preparing Data Step 4/4: One-Hot Encoding')
    month_delays = pd.get_dummies(month_delays)
    
    return month_delays, month_label


def calcAccuracy(Y_true, Y_preds):  
    accScore = 0
    numRows = Y_true.shape[0]
    
    Y_true = Y_true.values
    
    for i in range(numRows):
        if Y_true[i] == Y_preds[i]:
            accScore += 1
    
    return (accScore/numRows) * 100


def calcMetrics(Y_true,Y_preds):
    print()
    print('Metrics From the Data')
    print("Accuracy of Predictions is:", calcAccuracy(Y_true,Y_preds))
    print('Average Precision Score:', metrics.average_precision_score(Y_true,Y_preds))
    print('F1-Score:', metrics.f1_score(Y_true,Y_preds, average = None))
    print('Precision Score:', metrics.precision_score(Y_true,Y_preds, average = None))
    print('Recall Score:', metrics.recall_score(Y_true,Y_preds, average = 'binary'))


"""
Returns top 50
airports by flights
per day
"""
def top_50_airports():
    print('Top 50 Airports Step 1/2: Pulling Data From PopSQL')   
    CONNECTION_STRING = ""
    SQLconn = psycopg2.connect(CONNECTION_STRING)
    SQLcursor = SQLconn.cursor()
    
    # we will select the airports with top 50 num '
    # avg flights per day
    SQLcursor.execute(
        """select avg(num_flights) as avg_num_flights,
                  origin, avg(lat) as lat, avg(lon) as lon
           from
                (select count(1) as num_flights, origin,
                    avg(origin_lat) as lat,
                    avg(origin_long) as lon
                from plane_delays.y2018
                group by fl_date, origin) as iq 
            group by origin
            order by 1;"""
        )
    
    avg_flights_per_day = SQLcursor.fetchall()
    print('Top 50 Airports Final Step: Creating Dataframe')
    avg_flights_per_day = pd.DataFrame(avg_flights_per_day)
        # airports with top 50 avg flights per day
    avg_flights_per_day = (avg_flights_per_day.rename(
        columns={
               0: 'avg_flights',
               1: 'ORIGIN',
               2: 'LAT',
               3: 'LONG'
               }
            ).sort_values('avg_flights', ascending=False).head(50))
        
    avg_flights_per_day.avg_flights = avg_flights_per_day.avg_flights.astype('int64')
    
    return avg_flights_per_day  

if __name__ == "__main__":
    base_dir = sys.argv[1]

    print("LOADING PANDAS")
    load_pandas(base_dir)
    
    print("LOADING SQL")
    load_sql(base_dir)
    
    print("PERFORMING TESTS")
    tests(base_dir)
    
    print("CREATING PLOTS")
    make_plots(base_dir)

    print("PERFORMING ANALYSIS")
    perform_NBayes_analysis()

    print("PERFORMING LOG REG ANALYSIS")
    perform_LogReg_analysis()
