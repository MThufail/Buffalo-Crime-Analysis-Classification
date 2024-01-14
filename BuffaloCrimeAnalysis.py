#!/usr/bin/env python
# coding: utf-8

# ##### Group 48	Mahammad Thufail, Rekha Anvitha Inturi, Sai Shirini Prathigadapa

# # Buffalo Crime Analysis

# #### Data Parsing, Database Normalization, SQL Querying, Data Analysis and ML Classification

# ## Motivation

# #### Analyzing different types of incidents in Buffalo helps us understand what kinds of things are happening in the city. By sorting and organizing these incidents into groups, we can learn more about them. This helps the police and other people who work to keep us safe figure out how to best deal with each kind of incident.

# ### Utility Functions

# In[1]:


import pandas as pd
import sqlite3
from sqlite3 import Error

# Creates a connection to an SQLite database file.
def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn

# Creates a table in the SQLite database based on the provided SQL schema.
def create_table(conn, create_table_sql, drop_table_name=None):

    if drop_table_name: # You can optionally pass drop_table_name to drop the table.
        try:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as e:
            print(e)

    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

# Executes an SQL statement and fetches all the resulting rows.
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows


# ### Data Parsing

# In[2]:


# Creates multiple tables in the SQLite database to store different types of information related to incidents and census data.
def create_tables(conn):

    # INCIDENT INFO TABLE
    # Table storing information related to incidents, including date, type, location, and description.

    create_table(conn,
    """
    CREATE TABLE IF NOT EXISTS INCIDENT_INFO(
        INCIDENT_ID INT PRIMARY KEY,
        INCIDENT_DATETIME DATETIME,
        INCIDENT_TYPE_PRIMARY TEXT,
        INCIDENT_DESCRIPTION TEXT,
        PARENT_INCIDENT_TYPE TEXT,
        HOUR_OF_DAY INT,
        DAY_OF_WEEK TEXT,
        ADDRESS TEXT,
        CITY TEXT,
        STATE TEXT,
        LOCATION POINT,
        LATITUDE INT,
        LONGITUDE INT,
        Neighborhood TEXT);""",
    drop_table_name='INCIDENT_INFO')

    # CENSUS 2010 INFO TABLE
    # Table containing census information specific to 2010, including tract details and block information.

    create_table(conn,
    """
    CREATE TABLE IF NOT EXISTS CENSUS_INFO_2010(
       CENSUS_TRACT_2010 TEXT NOT NULL PRIMARY KEY,
       CENSUS_BLOCKGROUP_2010 TEXT,
       CENSUS_BLOCK_2010 TEXT );""",
    drop_table_name='CENSUS_INFO_2010')

    # CENSUS INFO TABLE
    # Table storing general census information, including tract, block, district, and zip code details.

    create_table(conn, """
    CREATE TABLE IF NOT EXISTS CENSUS_INFO(
        CENSUS_TRACT TEXT NOT NULL PRIMARY KEY,
        CENSUS_BLOCK TEXT,
        CENSUS_BLOCKGROUP TEXT,
        TRACTCE20 TEXT,
        POLICE_DISTRICT TEXT,
        COUNCIL_DISTRICT TEXT,
        ZIP_CODE INT
    );
    """, drop_table_name='CENSUS_INFO')

    # GEOID INFO TABLE
    # Table containing geographical information using GEOID identifiers for tracts, blocks, and block groups.

    create_table(conn, """
    CREATE TABLE IF NOT EXISTS GEOID_INFO(
        GEOID20_TRACT TEXT NOT NULL PRIMARY KEY,
        GEOID20_BLOCKGROUP TEXT,
        GEOID20_BLOCK TEXT
    );
    """, drop_table_name='GEOID_INFO')

    # CASE INFO TABLE
    # Table linking incident data with census and geographical information for case references.

    create_table(conn, """
    CREATE TABLE IF NOT EXISTS CASE_INFO(
        CASE_NUMBER TEXT NOT NULL PRIMARY KEY,
        INCIDENT_ID INT NOT NULL,
        CREATED_AT DATETIME,
        UPDATED_AT DATETIME,
        CENSUS_TRACT_2010 TEXT NOT NULL,
        CENSUS_TRACT TEXT NOT NULL,
        GEOID20_TRACT TEXT NOT NULL,
    FOREIGN KEY (INCIDENT_ID) REFERENCES INCIDENT_INFO(INCIDENT_ID),
    FOREIGN KEY (CENSUS_TRACT_2010) REFERENCES CENSUS_INFO_2010(CENSUS_TRACT_2010),
    FOREIGN KEY (CENSUS_TRACT) REFERENCES CENSUS_INFO(CENSUS_TRACT),
    FOREIGN KEY (GEOID20_TRACT) REFERENCES GEOID_INFO(GEOID20_TRACT)
    );
    """ , drop_table_name='CASE_INFO')

# Create a connection to the SQLite database and create tables
conn = create_connection('BuffaloCrim.db',delete_db=True)
create_tables(conn)


# ### Getting Data

# In[3]:


# Reads and processes data from a CSV file, filtering and preparing it for storage or further analysis.
def retrievedata(data_filename, normalized_database_filename):
    conn = create_connection(normalized_database_filename)
    header = None
    lines = []
    i = 1
    with open(data_filename) as file:
        for line in file:
            if i <= 300000:
                if not line.strip():
                    continue
                elif header==None:
                    header = line.strip().split(',')
                else:
                    line += f",{i}"
                    if line.strip().split(",")[16].strip() and line.strip().split(",")[19].strip() and line.strip().split(",")[26].strip():
                        lines.append(line.strip().split(",") )
                        i += 1
                    else:
                        continue
            else :
                break
    return lines

# Example usage of the retrievedata function
lines = retrievedata('Crime_Incidents.csv','normalized.db')
lines


# ###  Data Normalization

# In[4]:


from datetime import datetime

# Function to insert data into the INCIDENT_INFO table
def INSERT_INCIDENT_INFO(conn, values):
        cur = conn.cursor()
        cur.execute("""INSERT or Ignore INTO INCIDENT_INFO VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?);""", values)
        return cur.lastrowid

# Function to process and insert data from 'lines' into the INCIDENT_INFO table
def PUT_INCIDENT_INFO(conn,lines):
    with conn:
        i=1
        for line in lines:
            INCIDENT_DATETIME=line[1]
            if line[1]!='':
                input_datetime_str = line[1]
                input_datetime = datetime.strptime(input_datetime_str, '%Y-%m-%dT%H:%M:%S')
                INCIDENT_DATETIME = input_datetime.strftime('%m/%d/%Y %I:%M:%S %p')
            INCIDENT_ID = i
            INCIDENT_TYPE_PRIMARY = line[3]
            INCIDENT_DESCRIPTION = line[4]
            PARENT_INCIDENT_TYPE = line[5]
            HOUR_OF_DAY = line[6]
            DAY_OF_WEEK = line[7]
            ADDRESS = line[8]
            CITY = line[9]
            STATE = line[10]
            LOCATION = line[11]
            LATITUDE = line[12]
            LONGITUDE = line[13]
            Neighborhood = line[22]
            i+=1
            INSERT_INCIDENT_INFO(conn,( INCIDENT_ID, INCIDENT_DATETIME, INCIDENT_TYPE_PRIMARY, INCIDENT_DESCRIPTION, PARENT_INCIDENT_TYPE, HOUR_OF_DAY,
            DAY_OF_WEEK, ADDRESS, CITY, STATE, LOCATION, LATITUDE, LONGITUDE,Neighborhood))

# Call the function to insert data into the INCIDENT_INFO table
PUT_INCIDENT_INFO(conn,lines)
df_final= pd.read_sql_query("""SELECT * FROM INCIDENT_INFO""", conn)
df_final


# In[5]:


# Function to insert data into the CENSUS_INFO_2010 table
def INSERT_CENSUS_INFO_2010(conn, values):
        cur = conn.cursor()
        cur.execute("""INSERT INTO CENSUS_INFO_2010 VALUES(?,?,?);""", values)
        return cur.lastrowid

# Function to process and insert data into the CENSUS_INFO_2010 table
def PUT_CENSUS_INFO_2010(conn, lines):
    census_tract = []
    with conn:
        i=1
        for line in lines:
            INCIDENT_ID = i
            CENSUS_TRACT_2010 = line[16]
            CENSUS_BLOCKGROUP_2010 = line[17]
            CENSUS_BLOCK_2010 = line[18]
            i+=1
            if CENSUS_TRACT_2010 not in census_tract and CENSUS_TRACT_2010.strip():
                census_tract.append(CENSUS_TRACT_2010)
                INSERT_CENSUS_INFO_2010(conn,(CENSUS_TRACT_2010, CENSUS_BLOCKGROUP_2010, CENSUS_BLOCK_2010))
            else:
                continue

# Call the function to insert data into the CENSUS_INFO_2010 table
PUT_CENSUS_INFO_2010(conn, lines)
df = pd.read_sql_query("""SELECT * FROM CENSUS_INFO_2010""", conn)
df


# In[6]:


# Function to insert data into the CENSUS_INFO table
def INSERT_CENSUS_INFO(conn, values):
        cur = conn.cursor()
        cur.execute("""INSERT INTO CENSUS_INFO VALUES(?,?,?,?,?,?,?);""", values)
        return cur.lastrowid

# Function to process and insert data into the CENSUS_INFO table
def PUT_CENSUS_INFO(conn, lines):
    census_tract = []
    with conn:
        for line in lines:
            CENSUS_TRACT = line[19]
            CENSUS_BLOCK = line[20]
            CENSUS_BLOCKGROUP = line[21]
            TRACTCE20 = line[24]
            POLICE_DISTRICT = line[23]
            COUNCIL_DISTRICT = line[29]
            ZIP_CODE=line[28]


            if CENSUS_TRACT not in census_tract:
                if CENSUS_TRACT.strip():
                    census_tract.append(CENSUS_TRACT)
                    INSERT_CENSUS_INFO(conn,(CENSUS_TRACT, CENSUS_BLOCK, CENSUS_BLOCKGROUP, TRACTCE20
                                                  , POLICE_DISTRICT, COUNCIL_DISTRICT,ZIP_CODE))
                else: continue
            else:
                continue

# Call the function to insert data into the CENSUS_INFO table
PUT_CENSUS_INFO(conn, lines)
df = pd.read_sql_query("""SELECT * FROM CENSUS_INFO;""", conn)
df


# In[7]:


# Function to insert data into the GEOID_INFO table
def INSERT_GEOID_INFO(conn, values):
        cur = conn.cursor()
        cur.execute("""INSERT INTO GEOID_INFO VALUES(?,?,?);""", values)
        return cur.lastrowid

# Function to process and insert data into the GEOID_INFO table
def PUT_GEOID_INFO(conn, lines):
    GEOID= []
    with conn:
        for line in lines:
            GEOID20_TRACT = line[25]
            GEOID20_BLOCKGROUP = line[26]
            GEOID20_BLOCK = line[27]

            if GEOID20_TRACT not in GEOID:
                if GEOID20_TRACT.strip():
                    GEOID.append(GEOID20_TRACT)
                    INSERT_GEOID_INFO(conn,(GEOID20_TRACT, GEOID20_BLOCKGROUP, GEOID20_BLOCK))
                else:
                    continue
            else:
                continue

# Call the function to insert data into the GEOID_INFO table
PUT_GEOID_INFO(conn, lines)
df = pd.read_sql_query("""SELECT * FROM GEOID_INFO""", conn)
df


# In[8]:


# Function to insert data into the CASE_INFO table
def INSERT_CASE_INFO(conn, values):
        cur = conn.cursor()
        cur.execute("""INSERT OR IGNORE INTO CASE_INFO VALUES(?,?,?,?,?,?,?);""", values)
        return cur.lastrowid

# Function to process and insert data into the CASE_INFO table
def PUT_CASE_INFO(conn, lines):
    with conn:
        i=1
        for line in lines:
            CASE_NUMBER  = line[0]
            INCIDENT_ID = i
            CREATED_AT=line[14]
            if line[14]!='':
                input_datetime_str = line[14]
                input_datetime = datetime.strptime(input_datetime_str, '%Y-%m-%dT%H:%M:%S')
                CREATED_AT = input_datetime.strftime('%m/%d/%Y %I:%M:%S %p')
            UPDATED_AT = line[15]
            if line[15]!='':
                input_datetime_str = line[15]
                input_datetime = datetime.strptime(input_datetime_str, '%Y-%m-%dT%H:%M:%S')
                CREATED_AT = input_datetime.strftime('%m/%d/%Y %I:%M:%S %p')
            CENSUS_TRACT_2010 = line[16]
            CENSUS_TRACT = line[19]
            GEOID20_TRACT = line[25]
            i+=1
            INSERT_CASE_INFO(conn,(CASE_NUMBER, INCIDENT_ID, CREATED_AT, UPDATED_AT, CENSUS_TRACT_2010, CENSUS_TRACT, GEOID20_TRACT))

# Call the function to insert data into the CASE_INFO table
PUT_CASE_INFO(conn, lines)
normal_df = pd.read_sql_query("""SELECT * FROM CASE_INFO""", conn)
normal_df


# ### SQL Querying and Data Visualization

# In[9]:


# Extracting Yearly Incident Counts from Database and Filtering Post-2000
df = pd.read_sql_query("""
    SELECT
    SUBSTRING(INCIDENT_DATETIME,7,4) as "Year",
    COUNT(INCIDENT_ID) as COUNT
    FROM INCIDENT_INFO
    WHERE CAST(YEAR AS INT) > 2000
    GROUP BY YEAR
    ORDER BY YEAR;
 """, conn)
df


# In[10]:


# Visualizing Yearly Crime Trends: Interactive Plot of Incident Counts
import pandas as pd
import plotly.express as px

# Assuming 'df' is your DataFrame
# Replace column names with your actual column names

# Plotting with Plotly
fig = px.bar(df, x='Year', y='COUNT', color='COUNT', title='Number of Crimes per Year',
             labels={'COUNT': 'Number of Crimes', 'Year': 'Year'},
             color_continuous_scale='Blues')

# Add interactive features
fig.update_layout(
    xaxis=dict(type='category'),
    xaxis_title='Year',
    yaxis_title='Number of Crimes',
    coloraxis_colorbar=dict(title='Number of Crimes'),
    template='plotly_dark'
)

# Show the plot
fig.show()


# In[11]:


# Aggregated Incident Counts by Parent Incident Type: Analysis Result
df = pd.read_sql_query( """
SELECT
    PARENT_INCIDENT_TYPE,
    COUNT(INCIDENT_ID) as NUMBER_OF_INCIDENTS
FROM INCIDENT_INFO
GROUP BY PARENT_INCIDENT_TYPE;
 """, conn)
df


# In[12]:


# Visualizing Incident Type Distribution: Interactive Pie Chart
import plotly.express as px
import pandas as pd

# Sample data (replace this with your DataFrame or data)
data = {
    "Incident_Type": df['PARENT_INCIDENT_TYPE'],
    "Number_of_Incidents":df['NUMBER_OF_INCIDENTS']
}

# Creating a DataFrame from the sample data (replace this with your DataFrame)
df = pd.DataFrame(data)

# Creating an interactive pie chart
fig = px.pie(df, values='Number_of_Incidents', names='Incident_Type', title='Incident Type Distribution')
fig.show()


# In[13]:


# Top Days of the Week by Incident Count: Analysis Result
df = pd.read_sql_query( """
SELECT
    DAY_OF_WEEK,
    COUNT(*) AS Incident_Count
FROM INCIDENT_INFO
GROUP BY DAY_OF_WEEK
ORDER BY Incident_Count DESC;

 """, conn)
df
df=df.iloc[:7]
df


# In[14]:


# Visualizing Incident Counts Across Days of the Week: Interactive Line Chart
import plotly.express as px
import pandas as pd

# Sample data (replace this with your DataFrame or data)
data = {
    "Day_of_Week": df['DAY_OF_WEEK'],
    "Incident_Count": df['Incident_Count']
}

# Creating a DataFrame from the sample data (replace this with your DataFrame)
df = pd.DataFrame(data)

# Creating an interactive line chart
fig = px.line(df, x='Day_of_Week', y='Incident_Count', title='Incident Counts by Day of the Week')
fig.update_xaxes(type='category')  # Ensure the x-axis treats values as categories
fig.show()


# In[15]:


# Hourly Incident Distribution: Analysis Result
df = pd.read_sql_query( """
SELECT
    HOUR_OF_DAY,
    COUNT(INCIDENT_ID) AS NO_OF_INCIDENTS
FROM INCIDENT_INFO
GROUP BY HOUR_OF_DAY
ORDER BY  HOUR_OF_DAY

 """, conn)
df


# In[16]:


# Hourly Incident Distribution: Interactive Scatter Plot
import plotly.express as px
import pandas as pd

# Sample data
data = {
    "HOUR_OF_DAY":df['HOUR_OF_DAY'],
    "NO_OF_INCIDENTS": df['NO_OF_INCIDENTS']
}

# Creating a DataFrame from the sample data
df = pd.DataFrame(data)

# Creating an interactive scatter plot
fig = px.scatter(df, x='HOUR_OF_DAY', y='NO_OF_INCIDENTS', title='Incidents by Hour of the Day')
fig.update_traces(mode='markers', marker=dict(size=10))  # Customizing marker size
fig.update_xaxes(type='category')  # Ensure the x-axis treats values as categories
fig.show()


# In[17]:


# "Incident Count Distribution by Hour: Grouped Bar Chart"
import plotly.express as px
import pandas as pd

# Sample data
data = {
     "HOUR_OF_DAY":df['HOUR_OF_DAY'],
    "NO_OF_INCIDENTS": df['NO_OF_INCIDENTS']
}

# Creating a DataFrame from the sample data
df = pd.DataFrame(data)

# Creating a grouped bar chart
fig = px.bar(df, x='HOUR_OF_DAY', y='NO_OF_INCIDENTS',
             title='Incidents by Hour of the Day (Grouped Bar Chart)')
fig.update_layout(xaxis=dict(type='category', title='Hour of the Day'))
fig.show()


# In[18]:


# Analysis of Incident Counts by Council District: Census Data Comparison
df = pd.read_sql_query( """
SELECT
    CENSUS_INFO.COUNCIL_DISTRICT,
    COUNT(CASE_INFO.INCIDENT_ID) AS Incident_Count
FROM CENSUS_INFO
LEFT JOIN CASE_INFO ON CENSUS_INFO.CENSUS_TRACT = CASE_INFO.CENSUS_TRACT
GROUP BY CENSUS_INFO.COUNCIL_DISTRICT
ORDER BY CENSUS_INFO.COUNCIL_DISTRICT;

 """, conn)
df


# In[19]:


# Incident Counts by Council District: Visualized Treemap
import plotly.express as px
import pandas as pd

# Sample data
data = {
    "COUNCIL_DISTRICT": df['COUNCIL_DISTRICT'],
    "Incident_Count":df['Incident_Count']
}

# Creating a DataFrame from the sample data
df = pd.DataFrame(data)

# Creating an interactive treemap
fig = px.treemap(df, path=['COUNCIL_DISTRICT'], values='Incident_Count',
                 title='Incident Counts by Council District (Treemap)')
fig.show()


# ### Data Preprocessing

# In[20]:


df=df_final.copy()


# In[21]:


df


# In[22]:


# Converting Longitude and Latitude to String Data Type

df['LONGITUDE'] = df['LONGITUDE'].astype(str)
df['LATITUDE'] = df['LATITUDE'].astype(str)


# In[23]:


# Label Encoding Applied to Categorical Columns

from sklearn.preprocessing import LabelEncoder
df['INCIDENT_DESCRIPTION'] = LabelEncoder().fit_transform(df['INCIDENT_DESCRIPTION'])
df['ADDRESS'] = LabelEncoder().fit_transform(df['ADDRESS'])
df['INCIDENT_TYPE_PRIMARY'] = LabelEncoder().fit_transform(df['INCIDENT_TYPE_PRIMARY'])
df['PARENT_INCIDENT_TYPE'] = LabelEncoder().fit_transform(df['PARENT_INCIDENT_TYPE'])
df['DAY_OF_WEEK']=LabelEncoder().fit_transform(df['DAY_OF_WEEK'])
df['LONGITUDE']=LabelEncoder().fit_transform(df['LONGITUDE'])
df['LOCATION']=LabelEncoder().fit_transform(df['LOCATION'])
df['LATITUDE']=LabelEncoder().fit_transform(df['LATITUDE'])
df['Neighborhood']=LabelEncoder().fit_transform(df['Neighborhood'])


# In[24]:


# Removing Columns: CITY, STATE, INCIDENT_DATETIME

df=df.drop('CITY',axis=1)
df=df.drop('STATE',axis=1)
df=df.drop('INCIDENT_DATETIME',axis=1)


# In[25]:


df


# In[27]:


# Feature Selection using Heatmap

import matplotlib.pyplot as plt
import seaborn as sns
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# In[28]:


# Removing Columns: LATITUDE, LONGITUDE
df=df.drop('LATITUDE',axis=1)
df=df.drop('LONGITUDE',axis=1)


# ### ML Classification

# In[29]:


# Splitting of Predictor and Target Variables
X = df.drop('PARENT_INCIDENT_TYPE', axis=1)
Y=df['PARENT_INCIDENT_TYPE']


# In[30]:


#  Heatmap of Predictors
correlation_matrix = X.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# In[31]:


# Applying Decision Tree Classifier and Finding Accuracy

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[32]:


# Classification Report Generated for Decision Tree Classifier

class_report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
class_report


# In[33]:


# Confusion Matrix Calculated for Decision Tree Classifier

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[34]:


Y1=df_final['PARENT_INCIDENT_TYPE']
Y1.values


# In[35]:


Y=df['PARENT_INCIDENT_TYPE']
Y.values


# In[36]:


PARENT_INCIDENT_TYPE={}
for i,j in zip(Y1.values, Y.values):
    PARENT_INCIDENT_TYPE[j]=i
sorted_PARENT_INCIDENT_TYPE = {k: v for k, v in sorted(PARENT_INCIDENT_TYPE.items())}
sorted_PARENT_INCIDENT_TYPE


# ## Conclusion

# #### In wrapping up our study, we've successfully used a decision tree to sort different types of incidents in Buffalo. The model did well with most categories, but there's room for improvement, especially in dealing with incidents like 'Other Sexual Offense', 'Sexual Assault' and 'Sexual Offense'. Getting better at predicting these serious incidents can help the police plan more effectively. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




