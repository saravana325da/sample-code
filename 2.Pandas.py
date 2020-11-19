# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:39:46 2020

@author: SArumug1
"""
#Pandas tips
https://towardsdatascience.com/30-examples-to-master-pandas-f8a2da751fa4 
 
#folder
import os
os.getcwd()
os.chdir('C:\\Users')
os.listdir()
os.listdir('D:\\')
os.mkdir('C:\\Users\\test1')
os.rename('test','new_one')
os.remove('old.txt') #The rmdir() method can only remove empty directories.

-----
import pandas as pd
from pandas import *
pip install pandasgui

pd.options.display.max_columns =40
pd.options.display.max_rows =40
-----------

# File Handling - choose a file and load to pandas
import os
import re
folder = r'C:\Users\hp\Documents\Python\1.Python Practice\Python certification Systech\Systech Python certification internal May20'
def rename_files():
    file_list = os.listdir(folder)
    
    for file in file_list:
        
        if file.startswith('class_5') and file.endswith('t'):
            df111 = pd.read_csv(r'C:\Users\hp\Documents\Python\1.Python Practice\Python certification Systech\Systech Python certification internal May20\class_5_report.txt')
        
        if file.startswith('class_6') and file.endswith('t'):
            df122 = pd.read_csv(r'C:\Users\hp\Documents\Python\1.Python Practice\Python certification Systech\Systech Python certification internal May20\class_6_report.txt')

----------

%%time
 
#read csv files
df2=pd.read_csv(r'C:\Users\hp\Documents\Python\1.Python Practice\Fox python evaluation\BTN.csv',
                sep=',', skiprows=32,nrows=17,usecols=(0,1,2,3),
                dtype={'Twitter': 'Int64','Facebook': 'Int64','Instagram': 'Int64'})
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",nrows=20,usecols=(1,2,3)) # load only 20 rows and only column 1,2,3 not 0th column/Index
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",header=None, low_memory=False) # if no header
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",sep=',',index=False, skiprows=2,header=None,names=['heading1','Heading2'])  # will give header name
col_names= ['Header1','Header2']
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",header=None,names=col_names)
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",header=1) # apply 2nd row as header
df = pd.read_csv("daily_show_guests.csv", na_values = ['NA', 'N/A','-']])
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",header=1,na_values=('n.a','not available'))
df=pd.read_csv("C:/Users/CAIA/Desktop/Book1.csv",header=1,na_values={
    'col1':['n.a','not available'],
    'col2':['n.a',-1]})
#url read as csv
other_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(other_path, header=None)
---
# if we use chunk size the result will be a result, we need to convert to df
df = read_csv('https://raw.githubusercontent.com/allisonhorst/palmerpenguins/1a19e36ba583887a4630b1f821e3a53d5a4ffb76/data-raw/penguins_raw.csv', chunksize = 10000)
df_list = []
for df in df:
    df_list.append(df)
df = pd.concat(df_list,sort=False)
---
-----

product_df.to_csv('aa.csv', index=False, # otherwise will add extra column at start
                  sep=',',
                  encoding='utf-8)
-------------
#read Excel
pandas.read_excel(io, sheet_name=0, header=0, names=None, index_col=None, parse_cols=None, 
                  usecols=None, squeeze=False, dtype=None, engine=None, converters=None, 
                  true_values=None, false_values=None, skiprows=None, nrows=None, na_values=None, 
                  keep_default_na=True, verbose=False, parse_dates=False, date_parser=None, 
                  thousands=None, comment=None, skip_footer=0, skipfooter=0, convert_float=True,
                  mangle_dupe_cols=True, **kwds)
df = pd.read_excel("C:/Users/CAIA/Desktop/sample.xlsx", sheet_name='Sheet1')
df = pd.read_excel("C:/Users/CAIA/Desktop/sample.xlsx", 1) #index 1  will get 2nd sheet,
#to excel output file
df.to_excel(file.xlsx,sheet_name='sheet1')
----
file = r'C:\Users\hp\Documents\Python\1.Python Practice\Data Wrangling\Data Wrangling Exercises Level 1\List_of_television_stations.xlsx'
-------# load all sheets to SQL table
df = pd.read_excel(file, sheet_name=None)
a=df.keys()

for j in a:
     df[j].to_sql(j,sql,index=False,if_exists='replace')

pd.read_sql_table('KB',sql)

---
#json load

import json
import csv
#companydata=json.loads(r'D:\OneDrive - Systech Solutions, Inc\sara\Learning\Tools\Python\Python Exersices_Saravanakumar Arumugam\Data Wrangling Exercises Level 1\emp_book_20191231.json')
#json.df = pd.DataFrame(companydata['products'])
-----

file=r'C:\Users\hp\Documents\Python\1.Python Practice\Data Wrangling\Data Wrangling Exercises Level 1\emp_book_20191231.json'
data = json.load(open(file))
fname1 = r'C:\Users\hp\Documents\Python\1.Python Practice\Data Wrangling\Data Wrangling Exercises Level 1\emp_book_20191231.csv'

with open(fname1, "w") as file:
    csv_file = csv.writer(file)
    csv_file.writerow(["ID","Name","Surname","Child Org","Phone","email","Address"])
    for item in data["Employees"]:
        csv_file.writerow([item['ID'],item['Name'],item['Surname'],item['Child Org'],item['Phone'],item['Email'],item['Address']])
 df1=pd.read_csv(fname1, sep=',')
df1.head()
--
import requests
req = requests.get('https://jsonplaceholder.typicode.com/users')
req #200 status code means that the request has succeeded 
users = req.json()
users[0]
type(users)
type(users[0])
# Saving loading data
import json
with open(‘json_write_test.json’, ‘w’) as f:
    json.dump(users, f, indent=4)
# read json
with open(‘json_write_test.json’) as f:
    data = json.load(f)

pd_data = pd.DataFrame(data)
pd_data.head()
----
# flatening field dict values to separate fields
def flatten_json(json):
    output = {}
 
    def flatten(inpt, name=’’):
        if type(inpt) is dict:
            for elem in inpt:
                flatten(inpt[elem], name + elem + ‘_’)
        elif type(inpt) is list:
            i = 0
            for elem in inpt:
                flatten(elem, name + str(i) + ‘_’)
                i += 1
        else:
            output[name[:-1]] = inpt
 
    flatten(json)
    return output
flattened = [flatten_json(row) for row in data]

---

-------
#json to csv/excel

outputcsv = (r'D:\OneDrive - Systech Solutions, Inc\sara\Learning\Tools\Python\Python Exersices_Saravanakumar Arumugam\Data Wrangling Exercises Level 1\jsontocsv.csv')
df=pd.read_json(r'D:\OneDrive - Systech Solutions, Inc\sara\Learning\Tools\Python\Python Exersices_Saravanakumar Arumugam\Data Wrangling Exercises Level 1\emp_book_20191231.json')
df.to_csv('outputcsv',sep=',',columns=['col1','col2'])
df.to_csv('outputcsv',sep=','drop(columns=['col1']))
df.to_excel(r'D:\OneDrive - Systech Solutions, Inc\sara\Learning\Tools\Python\Python Exersices_Saravanakumar Arumugam\Data Wrangling Exercises Level 1\jsontocsv.xlsx')
----
#Html file load
# pip install html5lib

import pandas as pd
from pandas import read_html
import html5lib

file = r'D:\OneDrive - Systech Solutions, Inc\sara\Learning\Tools\Python\Python Exersices_Saravanakumar Arumugam\Data Wrangling Exercises Level 1\List_of_television_stations.html'

tablelist = pd.io.html.read_html(file)
tablelist

table1=tablelist[1]

-----------

#html link # load data from Html link

import requests
import pandas as pd

url = 'copy link here with tables'
df=pd.read_html(url,header=0)
df[1] # give 2nd table
df[1][1:].columns # give column names
df[0][1:]['columnname']

-----------
#sampling Data sample
df_sample = df.sample(n=1000) #n: The number of rows in the sample
df_sample.shape

df_sample2 = df.sample(frac=0.1) #frac: The ratio of the sample size to the whole dataframe size
df_sample2.shape
-------------------
# Data Structures Series/Data frame
#list to df
product_data=[['e-book', 2000], ['plants', 6000], ['Pencil', 3000]] # 2d list, similar to 2d array
indexs=range(len(product_data))
columns_name=['product', 'unit_sold']
product_df = pd.DataFrame(data=product_data, index=indexs, columns=columns_name)
print(product_df)

#dictionaries to df

product_data={'product': ['e-book', 'plants', 'Pencil'], 'unit_sold': [2000, 5000, 3000]}
product_df = pd.DataFrame(data=product_data)
print(product_df)
-------
#print(type(df['name'])) # will be pandas.core.series (Each column will be series)

----
#Cleaning
import pandas as pd

file = r'C:\Users\hp\Documents\Python\1.Python Practice\datasets\Sample - Superstore_rowid.xls'
df=pd.read_excel(file)

---
# print df info
if len(df) > 0:
    print(f'Length of df {len(df)}, number of columns {len(df.columns)}, dimensions {df.shape}, number of elements {df.size}.')
else:
    print(f'Problem loading df, df is empty.')
---
#display
print(f"{pd.options.display.max_columns} columns")
print(f"{pd.options.display.max_rows} rows")

pd.options.display.max_columns = None
pd.options.display.max_rows = None

pd.options.display.max_columns = 50
pd.options.display.max_rows = 100


df.size # no of records
df.shape
df.info()
--
#memory reduce
df.info(memory_usage='deep')
print(df.memory_usage(deep=True))
print(df.memory_usage(deep=True).sum())


df.dtypes

df.head(5) #first 5 lines
#df[['patient_id','doctor_name','class']].head()
df.tail(10) #last 10 lines
-----
display(df.head())
display(df.tail())
-------
pd.set_option('display.max_columns', None) # display all columns
df.columns
df.columns.values
---
ndf = df.select_dtypes(include=[np.number]) # only numeric columns
ndf.head()

cdf=df.select_dtypes(exclude=[np.number]) # only category columns
cdf.head()
----
Column=list(df.columns)

#shown each columns and their unique values in dataset
for x in Column:
    print(x,df[x].unique())
    print()
    
df.index

df.nunique() #level of data
df.nunique(axis=0) #level of data
df['City'].nunique()
df['Region'].unique().sort()
print(df['Quantity'].sort_values().unique())
df['Sales'].size 
df['Sales'].values # return  output in array format
df['Sales'].value_counts() #no of records
month=df['Month'].values
#Datatype change/ #dtype / #convert dtype
df['Price'] = newDf['Price'].astype('int')
df['Geography'] = df['Geography'].astype('category') # object to category will reduce memory
df['Price'] = df.apply(lambda x: int(x['Price'].replace(',', '')),axis=1)

df2['Followers']=df2['Followers'].dt.strftime('%b-%y') #Apr-20 format
df['Date']=pd.to_datetime(df['Date']) # object to Datetime
df["JAAR"] = pd.to_datetime(df["JAAR"], format="%Y")
df['Order Date']=pd.to_datetime(df['Order Date'], format='%Y-%m-%d')

df['year']=pd.DatetimeIndex(df['Order Date']).year  #extract the year or the month using DatetimeIndex() method 
df['month']=pd.DatetimeIndex(df['Order Date']).month
 # Note: Pandas does not support dates before 1880, so we ignore these for this analysis
df["year"] = pd.to_datetime(df["year"], errors="coerce")
# convert to numeric
df2['Mark'] = pd.to_numeric(df2['Mark']) # string to int


data = data.astype({'age':np.float64,'fare':np.float64,'pclass':object})
df.Year = df.Year.astype('str')  #to string type
# if your your object column contains mixed data types, you can use 

df.col.apply(type).value_counts() to check
---
#format 
#Decimal for float value
df_new.round(1) #number of desired decimal points
pd.options.display.float_format = ‘{:.4f}’.format # 4 decimal places
f = 0.333333
print(f"this is f={f:.2f} rounded to 2 decimals")
# Output
this is f=0.33 rounded to 2 decimals
----
#display options
get_option: Returns what the current option is
set_option: Changes the option
max_colwidth: Maximum number of characters displayed in columns
max_columns: Maximum number of columns to display
max_rows: Maximum number of rows to display
    
pd.set_option("display.precision", 2)

-----

#describe fields 
df.describe() # only numeric fields (Mean,std,min,max,25,50,75)
df['Sales'].describe()

---
df.describe(include='all') # all fields (unique,frequency,top,first,last)
df.describe(include=['object']) # only categories data will describe
df[df['field']=='High'].dercribe()
df[(df['field']=='High') & (df.sales>=1000)].dercribe()

df.describe(include='all')
display(df.describe(include=['category'])) # categorical types
display(df.describe(include=['number'])) # numerical types
#include=['category', 'object'] or exclude=['number']

#set index
 s = pd.Series(range(18249))  # like no of records in df.info()
df.set_index(s, inplace=True)
--
df['country'] = df.index
df = df.reset_index()
df = df.set_index('Gold')
df = df.set_index(['STNAME', 'CTYNAME'])
df = df.sort_index()

--
----

#Columns
df1 = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] #select required fields
df.rename(columns={'oldcol':'newcol','oldcol2':'newcol2'},inplace=True) # rename column names
df.columns = ['Rank', 'Title', 'Genre', 'Description', 'Director'] # to change all column names
---
df['education-num']=df['educational-num'] # rename educational-num to education-num
del df['educational-num']
---
df.drop(columns=['Postal Code'], inplace = True, axis = 1)
df_final=DataFrame(df,columns=['id','col1','col2','col3']) #keep required columns
keep_cols = ['Species', 'Region', 'Island', 'Culmen Length (mm)', 'Culmen Depth (mm)', 'Flipper Length (mm)', 'Body Mass (g)', 'Sex']
df = df.loc[:, keep_cols]	
to_drop = ['Edition Statement','Corporate Author','Corporate Contributors','Former owner']
df.drop(to_drop, inplace = True, axis = 1)
del.df['col1']

df.columns = ['col1','col2','col3'] # assign columns name to df

new_names =  {'Unnamed: 0': 'Country',
              '? Summer': 'Summer Olympics',
              '01 !': 'Gold',
              '02 !': 'Silver',
              '03 !': 'Bronze'}
             
olympics_df.rename(columns = new_names, inplace = True)

---
#The first parameter is the index of the location, the second one is the name of the column, and the third one is the value.
df_new.insert(0, 'Group', group) #insert col in sepecific locations
df_new
---
#column values
month=df['Month'].values # will assign the month field values to month variable
#maximum value of a column. When you take a single column you can think of it as a list and apply functions you would apply to a list. You can also use min for instance.

print(max(df['rating']))
----------
#Columns cleaning

df['Price'] = df.apply(lambda x: int(x['Price'].replace(',', '')),axis=1)
---
def clean_up(item):
    if '(' in item:
        return item[:item.find('(') - 1]
    
    if '[' in item:
        return item[:item.find('[')]
    

towns_df =  towns_df.applymap(clean_up)
-----------

 # remove unwanted charected in a field using apply function

unwanted_characters = ['[', ',', '-']

def clean_dates(dop):
    dop = str(dop)
    if dop.startswith('[') or dop == 'nan':
        return 'NaN'
    for character in unwanted_characters:
        if character in dop:
            character_index = dop.find(character)
            dop = dop[:character_index]
    return dop

df['Date of Publication'] = df['Date of Publication'].apply(clean_dates) 
-----
# Clean author field
def clean_author_names(author):
    
    author = str(author)
    
    if author == 'nan':
        return 'NaN'
    
    author = author.split(',')

    if len(author) == 1:
        name = filter(lambda x: x.isalpha(), author[0])
        return reduce(lambda x, y: x + y, name)
    
    last_name, first_name = author[0], author[1]

    first_name = first_name[:first_name.find('-')] if '-' in first_name else first_name
    
    if first_name.endswith(('.', '.|')):
        parts = first_name.split('.')
        
        if len(parts) > 1:
            first_occurence = first_name.find('.')
            final_occurence = first_name.find('.', first_occurence + 1)
            first_name = first_name[:final_occurence]
        else:
            first_name = first_name[:first_name.find('.')]
    
    last_name = last_name.capitalize()
    
    return f'{first_name} {last_name}'


df['Author'] = df['Author'].apply(clean_author_names)
------------
    def clean_title(title):

        if title == 'nan':
            return 'NaN'

        if title[0] == '[':
            title = title[1: title.find(']')]

        if 'by' in title:
            title = title[:title.find('by')]
        elif 'By' in title:
            title = title[:title.find('By')]

        if '[' in title:
            title = title[:title.find('[')]

        title = title[:-2]

        title = list(map(str.capitalize, title.split()))
        return ' '.join(title)

    df['Title'] = df['Title'].apply(clean_title)
    df.head()

-----
#calculated field / create #new column

df['col2']= df['col1'] # single column from df will make series
#add/addition to series
df3=df['col1']+df['col2'] # will +add both col 
#concatinate 2 column
df3=df['col1']+' '+df['col2']  # join both column

df['AvgRating'] = (df['Rating'] + df['Metascore']/10)/2
product_df['next_target'] = product_df['unit_sold'] + ( product_df['unit_sold'] * 10)/100']
#create column with title>=4 letters
new_df = df[df.apply(lambda x : len(x['Title'].split(" "))>=4,axis=1)]
#lambda
df.apply(lambda x: func(x['col1'],x['col2']),axis=1)
data['relatives'] = data.apply (lambda row: int((row['sibsp'] + row['parch']) > 0), axis=1)
----
def custom_rating(genre,rating):
    if 'Thriller' in genre:
        return min(10,rating+1)
    elif 'Comedy' in genre:
        return max(0,rating-1)
    else:
        return rating
        
df['CustomRating'] = df.apply(lambda x: custom_rating(x['Genre'],x['Rating']),axis=1)
--
#new file after filter records
#unique values from a column
unique_values = df['shift'].unique()
for value in unique_values:
    df1=df[df['shift']==value]
    output_filename='shift_'+str(value)+'training.xlsx'
    df1.to_excel(output_filename.xlsx, index=False)
---
#apply #progressbar
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()

df.progress_apply(lambda x: custom_rating_function(x['Genre'],x['Rating']),axis=1)
----

---------
#bin/#cut
df['Quantity'].unique()
df['bins']=pd.cut(df['Quantity'],bins=4,labels=[1,2,3,4])
pd.cut([0, 1, 1, 2], bins=4, labels=False)
pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)
pd.cut(np.array([1, 7, 5, 4, 6, 3]),
       3, labels=["bad", "medium", "good"])

----------
#preproccessing
#null Check
import numpy as np
df.replace('?',np.NaN,inplace=True)
df['Sex'].replace('.','Unknown', inplace=True)
df.isnull().sum() # display summary of null values per column
df.isna().sum()
df.isna().sum().sum()
df.isnull().values.any() # verify if any value is null
#print(df['Region'].isnull())
#print(df['Region'].notnull())
#df_nan will have the rows which has NaN values in them.
df_nan_row = df[df.isna()]
print(df.isna().sum())
df['Sex'].unique()

df.fillna(0,inplace=True)
df = df.fillna(0)
df = df.fillna(method='ffill') #fill missing values based on the previous or next value in a column (used in time series)
 
df[['NO','NO_2']] = df[['NO','NO_2']].fillna(0)

avg = df['Balance'].mean()
df['Balance'].fillna(value=avg, inplace=True) # fill null with mean 

df = df.replace(np.nan,'',regex=True)


df=df.dropna(axis = 1, how = 'all')
df.dropna(axis=0, how='any',  thresh=5, inplace=True) # <=4 missing values in a row will be droped

df = df[['sex', 'pclass','age','relatives','fare','survived']].dropna()
df.dropna(subset=["price"], axis=0)
----
#replace field value
df.replace('?','',inplace=True) # clean any column unwanted charector
df.replace({'male': 1, 'female': 0}, inplace=True)

df['A']=df['A'].replace(0,2)
df.loc[df['income']=='<=50K','income']=0  ##modified all the values of income field having "<=50K" to 0
df.replace([0, 1, 2, 3], 4)
#The values that fit the specified condition remain unchanged and the other values are replaced with the specified value.
df_new['Balance'] = df_new['Balance'].where(df_new['Group'] >= 6, 0)
----------
#duplicates/repeat records
# This shows rows that show up more than once and have the exact same column values. 
df[df.duplicated(keep = 'last')]

# # This shows all instances where pantient_id shows up more than once, but may have varying column values
df[df.duplicated(subset = 'patient_id', keep =False)].sort_values('patient_id')
df = df.drop_duplicates(subset = None, keep ='first')
df.drop_duplicates(subset='Name',keep='last') # column name 'Name'

--
#repeat records
repeat_patients = df.groupby(by = 'patient_id').size().sort_values(ascending =False)


filtered_patients = repeat_patients[repeat_patients > 2].to_frame().reset_index()
filtered_df = df[~df.patient_id.isin(filtered_patients.patient_id)]
filtered_df

# This is all the repeating patients details

df[df.patient_id.isin(filtered_patients.patient_id)]
------
#split  field values 
a='2020.0'
a.split('.')
a
firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
df.Year = [x.split('.')[0] for x in df.Year] # split only 2020 from 2020.0 
------

----------
#Merge Two DataFrames on Multiple Keys
print(pd.merge(left,right,on=['id','subject_id'],how='inner',indicator = True)) #Enabling indicator will provide information about the dataframe source of each row (left or right).
#Concate
print(pd.concat([one,two],axis=0))
movies_06_07 = pd.concat([movies_2006,movies_2007])
#concate all sheets into 1 df
df = pd.concat(pd.read_excel(workbook_url, sheet_name=None), ignore_index=True) # sheet_name= None read all sheets together.

-----------

#slice
#index Data selection method: slice data frame.
import numpy as np
# using numpy array
products = np.array([['','product','unit_sold'], [1, 'E-book', 2000],[2, 'Plants', 6000], [3, 'Pencil', 3000]])
product_pf = pd.DataFrame(data=products[1:,1:], # [1:,1:] from first row till end, from first column till end
                          index=products[1:,0], # [1:,0] from first row till end, only first column
                          columns=products[0,1:]) # [1:,0] only first row, form first column till end
print(product_pf) # output is same as of first case
-----------
row = product_df.iloc[2] # fetch third row
rows = product_df.iloc[0:3] # fetch rows from first till third but not third
product_df.at[2,'product]
---------
df.loc[10:15,['col1']] # specific rows and column

-----------------             
name=df(['name'])
print(name([0])) # will index the first element in the series
# get single value from df (indexing/slicing)
print(df.at(["rowlabel","column label"]))
print(df.iat([1,1])) # will give only one value
print(df.loc[[2,4,6],["Age":"Occupation"]) # will give range of df values
print(df.loc[df["field"]==True]) # will give all field filter by one field condition
print(df.loc[:,'name']) # same result as above
print(df.iloc[:,1]) # same result as above
df['Region']
----
x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
x['Christopher Brooks'] # Retrieve a value by using the indexing operator
---
#Filter the records
df_new[df_new.Names.str.startswith('Mi')] 
df_new[df_new.Names.str.endsswith('Mi')]
movies_2006 = df[df['Year']==2006] #filter records based on certain values

df3=(df.loc[df["Region"]=='South']) # will give all field filter by one field condition
# Single condition: dataframe with all movies rated greater than 8
df_gt_8 = df[df['Rating']>8]
# Multiple conditions: AND - dataframe with all movies rated greater than 8 and having more than 100000 votes
And_df = df[(df['Rating']>8) & (df['Votes']>100000)]
# Multiple conditions: OR - dataframe with all movies rated greater than 8 or having a metascore more than 80
Or_df = df[(df['Rating']>8) | (df['Metascore']>80)]
# Multiple conditions: NOT - dataframe with all movies rated greater than 8 or having a metascore more than 80 have to be excluded
Not_df = df[~((df['Rating']>8) | (df['Metascore']>80))] # Multiple conditions: NOT - dataframe with all emovies rated greater than 8 or having a metascore more than 90 have to be excluded
df[df['shipmode'].isin(['Regular','Air'])] # isin 
only_gold = df.where(df['Gold'] > 0)
only_gold = df[df['Gold'] > 0]
----
#Filter with query()

df.loc[(df['tip']>6) & (df['total_bill']>=30)]
df.query("tip>6 & total_bill>=30") #same result as above
new_df.query('Confirmed_New > Confirmed_Recovery').head()
cn_mean = new_df['Confirmed_New'].mean()
new_df.query('Confirmed_New > @cn_mean').head()
new_df.query('Text_Feature == "Blue"').head()
df2 = df.query('80000 < Balance < 100000')
# reference global variable name with @
median_tip = df['tip'].median()
display(df.query("tip>@median_tip").head())

# wrap column name containing . with backtick: `
df.rename(columns={'total_bill':'total.bill'}, inplace=True)
display(df.query("`total.bill`<20").head())
df.rename(columns={'total.bill':'total_bill'}, inplace=True)

# wrap string condition with single quotes (this is what I like)
display(df.query("day=='Sat'").head())
# could also do it the other way around (i.e. 'day=="Sat"')


--------
#delete records
#drop
df.drop(['row id','count'],axis=1,inplace=True) #inplace wil update the df,axis=1 is column
df.drop([2,3],axis=0,inplace=True) # axis=0 is Rows and 2,3 index value
df.drop(' ',axis=0,inplace=True) # axis=0 is Rows and 2,3 index value
table.drop(table.tail(1).index, inplace = True) # delete last row in pivot table
---




-----------
import datetime
#groupby 
df[['Geography','Gender','Exited']].groupby(['Geography','Gender']).mean()
df[['Geography','Gender','Exited']].groupby(['Geography','Gender']).agg(['mean','count'])

df_summary = df[['Geography','Exited','Balance']].groupby('Geography')\
.agg(
 Number_of_churned_customers = pd.NamedAgg('Exited', 'Sum'), # column heading customize
 Average_balance_of_customers = pd.NamedAgg('Balance', 'Mean')
)

df.groupby(list of columns to groupby on).aggregate({'colname':func1, 'colname2':func2}).reset_index()
df.groupby(['Category', 'Sub-Category']).agg({'Sales':'sum','Discount':'mean'}).reset_index()
df_new = df[['Geography','Exited','Balance']]\
.groupby(['Geography','Exited']).mean().reset_index()
---
import numpy as np
df.groupby(['Year','Genre']).aggregate({'Votes':np.sum, 'Rev_M':np.sum}).reset_index()
---
df.groupby('Category').Sales.sum() # category wise sum sales

df.groupby(['year','month'], sort=False).agg({'Sales':'sum'}) # no sorting
# max torque value grouped by profile id
grouped_df = df.groupby(['profile_id']).max()
print(grouped_df['torque'])

df.groupby(by =['class', 'doctor_name','clump_thickness']).size()
df.groupby(by = ['class']).size()

# How to view the data by aggeregting on more than one column
df.groupby('class').agg({'cell_size_uniformity': ['min', 'max'], 'normal_nucleoli': 'mean', 'class': 'count'})
# Find out the sum of votes and revenue by year
import numpy as np
df.groupby(['Year']).aggregate({'Votes':np.sum, 'Rev_M':np.sum}).reset_index()
df.groupby(['Year','Genre']).aggregate({'Votes':np.sum, 'Rev_M':np.sum}).reset_index()

piv = df.groupby(['field']).mean()
prductivity = piv.loc[:,"field":"field2"]
print(productivity)
---------
df.groupby('Sales Rep').agg({ 
    'Order Id':'size',
    'Val':['sum','mean'],
    'Sale':['sum','mean']
})
---------
#groupby multiple column
grouped_multiple = df.groupby(['LOB','PPG', 'PARTY_NAME', 'variable','FY_MONTH_ID']).agg({'value': ['sum']})
grouped_multiple.columns = ['sum']
grouped_multiple = grouped_multiple.reset_index()
print(grouped_multiple)
grouped_multiple.head()
grouped_multiple.to_excel(r'C:\Users\hp\Documents\Python\1.Python Practice\Iffco uplift value groupby.xlsx')
---
guest_group = guest_list_df.groupby('Group')
print(guest_group.get_group('Acting'))

--------
#Sorting
df.nsmallest(5, 'total_bill')
display(df.nlargest(5, 'total_bill'))
display(df.sort_values(by='total_bill', ascending=False).head())
df.sort_values(by=[‘total_bill’, ‘tip’], ascending=[True, False]).head() #ascending =[1,0] also same result

df.sales.sort_values()  # sort only column ascending
df.sales.sort_values(ascending=False).head() # sort column Decending
df['sales']sort_values()
df.sort_values('sales',ascending=False).head() # sort
df.sort_values(['sales','revenue'],ascending=False).head() # multiple fields
sorted_guest_df = guest_list_df.sort_values('GoogleKnowlege_Occupation', # sort by column
                                 ascending=False, # enable descending order
                                 kind='heapsort', #sorting algorithm
                                 na_position='last') # keep NaN value at last
--------
#rank
df_new['rank'] = df_new['Balance'].rank(method='first', ascending=False).astype('int')

df = pd.DataFrame(data={'Animal': ['cat', 'penguin', 'dog',
                                   'spider', 'snake'],
                        'Number_legs': [4, 2, 4, 8, np.nan]})
df
df['default_rank'] = df['Number_legs'].rank() #default
df['max_rank'] = df['Number_legs'].rank(method='max') # since ‘cat’ and ‘dog’ are both in the 2nd and 3rd position, rank 3 is assigned.)
df['NA_bottom'] = df['Number_legs'].rank(na_option='bottom') #NA also ranked at bottom
df['pct_rank'] = df['Number_legs'].rank(pct=True) #percentile rank.
df
-----
#Styling a dataframe
df_new.style.highlight_max(axis=0, color='darkgreen')

----
#statistic calc 

df['Sales','Discount'].describe()
df['Sales'].sum()

df['unit_sold'].mean()
df['unit_sold'].sum()
df.mean(axis=0) # axis 0 : columns, 1 Rows


------------------------
#EDA Exploratory analysis
df.groupby('category').sales.sum() # category wise sum sales
df.groupby('category').mean() # give avg value for all numeric column
df.groupby(['category','ship mode']).sales.agg(['sum','mean','max','min']) # all together
df.groupby('category').sales.sum().nlargest(5) # top5 products 
df.groupby('category').sales.sum().nsmallest(5) # bottom5 products 
df['productname'].value_counts() # will give top sold products counts
df['Productname'].nunique() # will shows distinct no of products
df.nunique() # distinct no of records
df.['productname'].unique() # distinct productname 
----
#crosstab
# cannot add index=True
pd.crosstab(df.Nationality,df.Hand)
pd.crosstab([df.index,df.Nationality],df.Hand)
pd.crosstab([df.Nationality,df.Sex],[df.Hand],margins=True) # index 2 coumns, and add row/columns total at the end
pd.crosstab([df.Sex],[df.Hand],normalize='index') 
import numpy as np 
pd.crosstab([df.Sex],[df.Hand],values=df.Age,aggfunc=np.average) # aggregate values avg
---
#melt makes columns to rows
reshaped_df = pd.melt(working_df,id_vars = ['Title','Rating','Votes','Rev_M'],value_vars = list(genre_set),var_name = 'Genre', value_name ='Flag')
reshaped_df  = reshaped_df[reshaped_df['Flag']==1]
#value_vars: List of vars we want to melt/put in the same column
#var_name: name of the column for value_vars
#value_name: name of the column for value of value_vars
-----

#pivot table
re_reshaped_df = reshaped_df.pivot_table(index=['Title','Rating','Votes','Rev_M'], columns='Genre', 
                        values='Flag', aggfunc='sum').reset_index()
re_reshaped_df=re_reshaped_df.fillna(0)
----
import numpy as np
table2 = pd.pivot_table(data = df, index = 'Genre', values = ['EU_Sales', 'JP_Sales'])
pd.pivot_table(df,'sales',index['col1'],columns['col2'],aggfunc=np.mean,margins=True,margins_name='Total') 
--

table4 = pd.pivot_table(df, index = ['Publisher', 'Genre'], 
                        aggfunc = {'EU_Sales':np.mean, 'Global_Sales':np.sum, 'JP_Sales':np.median})
table4.columns = ['EU_Sales_mean', 'Global_Sales_sum', 'JP_Sales_median']
---
table5 = pd.pivot_table(df, index = ['Year'], columns = ['Genre'], values = ['Global_Sales'],
                        aggfunc = np.sum, fill_value = 0)

---
#Calculating the percentage change through a column
ser = pd.Series([4,5,2,5,6,10])
ser.pct_change() #The change from the first element (4) to the second element (5) is %25 so the second value is 0.25.
pd.set_option("display.precision", 2)
---------
#string Method:
Splitting /Stripping /Replacing/Filtering/Combining
df_sample['col_a'].str.split(',')
A=df['col'].str.split().explode().reset_index(drop=True)#split
A.str.cat(sep=' ') #concatenates string
A.str.capitalize()
df_sample['state'] = df_sample['col_a'].str.split(',').str[1] # new col as state
categories.str.split('-', expand=True, n=2)
categories.str.rsplit('-', expand=True, n=2) #right side split
df_sample['col_b'].str.lstrip('$') #remove $ simbol
df_sample['col_d'] = df_sample['col_d'].str.strip() #leading and trailing spaces can be removed
df['category']= df['category'].str.upper() # converts to upper
df['category_lenth'] = df['category'].str.len().head() #lenth of all rows stored in new column
df['categ'].str.istitle().head() # check title
df['categ'].str.contains('Technology').head() 
df['catrg'].str.replace('NA','Nil ').head() 
df_sample[df_sample['col_a'].str.endswith('X')]
df_sample[df_sample['col_b'].str.startswith('$6')]
lower  = df_sample['col_b'].str[1:3]
upper  = df_sample['col_b'].str[-3:-1]
A.str[:2]
a.str.count('aa')
a.str.rstrip('123 ,')
df_sample['new']=df_sample['col_c'].str.cat(df_sample['col_d'], sep='-')#create a new column by concatenating “col_c” and “col_d” with “-” separator

#string literal
person = 'roman'
exercise = 0
print(f"{exercise}-times {person} exercised during corona epidemic")
# output
# 0-times Roman exercised during corona epidemic
--
print(f"{exercise+1}-times {person} exercised during corona epidemic")
# Output
# '1-times roman exercised during corona epidemic'
--


-------------
#regex regular expression:

import re    
pattern = "[0-9a-zA-Z]+@[a-zA-Z]+\.(com|in|net|edu)"

emailid =input('enter email id')
def email():
    
    if(re.search(pattern,emailid)):
        print("emailid is ok")
    else:
        print('emailid is not ok')
 
email()
-------------------
# replace phone no format
pattern = "(\d\d\d)-(\d\d\d)-(\d\d\d\d)"
newpattern = r"\1\2\3"
userinput='222-333-5555'
newinput=re.sub(pattern,newpattern,userinput)
print(newinput)

-----
#word Cloud
#https://towardsdatascience.com/one-line-of-python-code-to-help-you-understand-an-article-c8aacab33dcb
pip install wikipedia
pipinstall stylecloud

import stylecloud as sc
import wikipedia

sc.gen_stylecloud(
    file_path='data_architect.txt', 
    output_name='data_architect.png'
)
-----
sc.gen_stylecloud(
    file_path='covid-19.txt',
    icon_name='fas fa-viruses',
    output_name='covid-19.png'
)
----
sc.gen_stylecloud(
    text=wikipedia.summary("linkedin"),
    icon_name='fab fa-linkedin-in',
    output_name='linkedin.png'
)
-----
sc.gen_stylecloud(
    text=wikipedia.summary("twitter"),
    icon_name='fab fa-twitter',
    output_name='twitter.png'
)
----

--------------------
#Plot
----------
#viz
import hvplot
import hvplot.pandas
by_year.hvplot()
-----
#plots using Pandas
#https://towardsdatascience.com/an-ultimate-cheat-sheet-for-data-visualization-in-pandas-4010e1b16b5c

a = pd.Series([40, 34, 30, 22, 28, 17, 19, 20, 13, 9, 15, 10, 7, 3])
a.plot()
a.plot(figsize=(8, 6), color='green', title = 'Line Plot', fontsize=12)
a.plot(kind='area')
a.plot.area()

b = pd.Series([45, 22, 12, 9, 20, 34, 28, 19, 26, 38, 41, 24, 14, 32])
c = pd.Series([25, 38, 33, 38, 23, 12, 30, 37, 34, 22, 16, 24, 12, 9])
d = pd.DataFrame({'a':a, 'b': b, 'c': c})

d.plot.area(figsize=(8, 6), title='Area Plot')
d.plot.area(alpha=0.4, color=['coral', 'purple', 'lightgreen'],figsize=(8, 6), title='Area Plot', fontsize=12
#This .plot() can male eleven types of plots:
line
area
bar
barh
pie
box
hexbin
hist
kde
density
scatter
df.columns
df['BMXWT'].hist()
df2['Balance'].plot(kind='hist', figsize=(8,5))
df[['BMXWT', 'BMXHT', 'BMXBMI']].plot.hist(stacked=True, bins=20, fontsize=12, figsize=(10, 8))
df[['BMXWT', 'BMXHT', 'BMXBMI']].hist(bins=20,figsize=(10, 8))
df[['DMDEDUC2x', 'BPXSY1']].hist(by='DMDEDUC2x', figsize=(18, 12))
df.groupby('DMDMARTLx')['BPXSY1'].mean().plot(kind='bar', rot=45, fontsize=10, figsize=(8, 6)
df.groupby('DMDEDUC2x')['BPXSY1'].mean().plot(kind='barh', rot=45, fontsize=10, figsize=(8, 6))
#side by side bar
df_bmx.plot(x = 'RIDRETH1', 
            y=['BMXWT', 'BMXHT', 'BMXBMI'], 
            kind = 'bar', 
            color = ['lightblue', 'red', 'yellow'], 
            fontsize=10)
#stacked bar
df_bmx.plot(x = 'RIDRETH1', 
            y=['BMXWT', 'BMXHT', 'BMXBMI'], 
            kind = 'bar', stacked=True,
            color = ['lightblue', 'red', 'yellow'], 
            fontsize=10)                                              
#box
color = {'boxes': 'DarkBlue', 'whiskers': 'coral', 
         'medians': 'Black', 'caps': 'Green'}
df[['BMXBMI', 'BMXLEG', 'BMXARML']].plot.box(figsize=(8, 6),color=color)
df.head(300).plot(x='BMXBMI', y= 'BPXSY1', kind = 'scatter')
df.head(500).plot.scatter(x= 'BMXWT', y = 'BMXHT', c ='BMXLEG', s=50, figsize=(8, 6))
df.head(200).plot.scatter(x= 'BMXHT', y = 'BMXWT', 
                          s =df['BMXBMI'][:200] * 7, 
                          alpha=0.5, color='purple',
                         figsize=(8, 6))
df.plot.hexbin(x='BMXARMC', y='BMXLEG', C = 'BMXHT',
                         reduce_C_function=np.max,
                         gridsize=15,
                        figsize=(8,6))
#scatter Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df[['BMXWT', 'BMXHT', 'BMXBMI', 'BMXLEG', 'BMXARML']], alpha = 0.2, figsize=(10, 8), diagonal = 'kde')
#Probability distribution
df['BMXWT'].plot.kde()
df[['BMXWT', 'BMXHT', 'BMXBMI']].plot.kde(figsize = (8, 6))
#Bootstrap plot
from pandas.plotting import bootstrap_plot
bootstrap_plot(df['BMXBMI'], size=100, samples=1000, color='skyblue')




---

#############

from matplotlib import pyplot as plt
%rehashx
plt.plot()
plt.plot(df)
plt.show()
df['AveragePrice'].plot()
plt.bar(df)
df.plot.bar(legend = True)
df.plot.area()
df.plot.bar(stacked= True)
df.plot.bar()
df.plot.pie(subplots=True)
plt.plot(df)
plt.bar(df)
help(df.plot)

df.plot(kind='bar')
plt.show()
--------
#plotly
import plotly
import plotly.graph_objects as go

#scattorplot
data = [go.scattor(x=df['Date'],y=df['Profit'])]
fig = go.Figure(data)
fig.show()
plotly.offline.plot(fig,filename='"raph.html") # wxport graph to html file
-------------
#matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')
%matplotlib inline


t = table.plot(kind = 'line', figsize = (20, 10), lw = 2.5)
t.set_xlabel('Year', fontsize = 20)
t.set_ylabel('Sales (Millions)', fontsize = 20)
t.set_title('Sales by Year - Trends', fontsize = 20)
t.legend(fontsize = 20, edgecolor = 'black', facecolor = 'white', fancybox = True)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.tight_layout()                  
    -----------
# plot pie graph from income and peoples
distribution = np.array([Less_50K, Greater_50K])
labels = ['Below 50K', 'Above 50K']
explode = [0, 0.1]
plt.pie(x = distribution, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, startangle=45)
plt.show()

# plot histogram btw education num and people
plt.hist(x = Edu_data, bins = bins, color = 'green')
plt.xlabel('Education-Number')
plt.ylabel('People')
plt.show()

sns.distplot(Edu_data, hist = False, color = 'm')

#seaborn
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=5, figsize=(30,5))
sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])

--------
#to_sql
#sqlite
import sqlite3
from sqlalchemy import create_engine
sql=create_engine('sqlite://',echo=False)
table1.to_sql('table1',sql,index=False,if_exists='replace')
#results=sql.execute("Select * from table1")
pd.read_sql_table('table1',sql)
-----------------

#Sqlalchemy
from sqlalchemy import create_engine


SERVER = 'DESKTOP-4KM0D6G\SQLEXPRESS2'
DATABASE = 'tgt_saravana'
DRIVER = 'SQL Server Native Client 11.0'
USERNAME = 'sa'
PASSWORD = 'Systech123'
DATABASE_CONNECTION = f'mssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}?driver={DRIVER}'
sql = create_engine(DATABASE_CONNECTION)
connection = sql.connect()
connection.close()
-----
#to_sql 
df1.to_sql('Class_5_report', con=sql, if_exists='replace', index=False)                     
df3=pd.read_sql('Class_5_report',con=sql)
df3
------------
#from sql
import pandas as pd
DATABASE = 'northwind'
df3=pd.read_sql('Region',con=sql)
df3
df3.to_csv(r'C:\Users\hp\Documents\Python\1.Python Practice\Region.csv',index=False)
df4=pd.read_csv(r'C:\Users\hp\Documents\Python\1.Python Practice\Region.csv')
df4
df4.to_sql('Region1',sql,index=False,if_exists='replace')
--
# Example code: f-string's and variables triple quote allow us to use single quote inside the querry
region = tuple('Anvers')
df = read_sql(f'''SELECT * FROM penguins WHERE Region IN {region} AND Date Egg > '2007-11-11' ''', con)

----
#sQL Datasource read
import MySQLdb
from pandas import DataFrame
from pandas.io.sql import read_sql

db = MySQLdb.connect(host="localhost",    # your host, usually localhost
                     user="root",         # your username
                     passwd="password",   # your password
                     db="dbname")         # name of the data base

query = "SELECT * FROM tablename"

df = read_sql(query, db)
---
my_query = "SELECT * FROM CURRENT_TABLE"
sql_data = pandas.read_sql(my_query, connection)
---
#to_aws : AWS SDK for Python, boto3




--------------
#EDA/#eda/#Exploratory Data Analysis
----
pip install pandas-profiling
import pandas_profiling
prof_report = pandas_profiling.ProfileReport(df , title = 'Titanic Report')
prof_report.to_widgets()
prof_report.to_html()
------
pip install sweetviz
import sweetviz
import pandas as pd
df = pd.read_csv('train.csv')
report = sweetviz.analyze(df)
report.show_html()

report.show_html("Titanic.html")  #You can also pass a file name to show_html()
-----
pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
df = AV.AutoViz('train.csv')

-----
#profiling
from pathlib import Path
from pandas_profiling import ProfileReport
df = pd.read_csv(r'C:\Users\hp\Documents\GitHub\pandas-profiling\examples\bank_marketing_data\file.csv')
profile = ProfileReport(
        df, title="Profile Report of the UCI Bank Marketing Dataset", explorative=True)
--
   profile = ProfileReport(
        df,
        title="NASA Meteorites",
        html={"style": {"logo": logo_string}},
        correlations={"cramers": {"calculate": False}},
        explorative=True)

profile.to_file(Path("uci_bank_marketing_report.html"))

--

-----
#We can specify <pdb.set_trace()> anywhere in the script and set a breakpoint there.
 It’s extremely convenient.
import pdb
pdb.set_trace()


--------
#Check The Memory Usage Of An Object.
df.memory_usage()

import sys 
x = 1
print(sys.getsizeof(x))