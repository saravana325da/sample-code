# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:45:15 2020

@author: hp
"""
----------
# =============================================================================
# #Spyder help

keep curser on pandas and press ctrl+i to get help topics
# =============================================================================
import pandas as pd
import requests
#debugger

import pdb
pdb.set_trace()
--------------

# Installing Packages

python -m ensurepip --default-pip #install pip
python --version # 3.7 
#venv
-------
To Install requirements using Conda Environments and Python3
$ conda env create -f data_wrangling_3.yml
$ source activate conda_dw3
---
python --version  # to get python version
--version

---------------
#spyder
pip --version
in anaconda editor 
conda update anaconda
conda update --all
conda update -n base -c defaults conda
conta update spyder
conda install spyder=4.1.5

-------------------------------
# =============================================================================
# # Jupyter hint 
# =============================================================================
shift+tab --> will show the help
Tab --> Auto complete

#Package update
pip install jupyter notebook # install jupyter notebook in cmd prompt
pip install notebook --upgrade 
jupyter notebook # to open browser window for notebook
--
# conda
conda install dtale -c conda-forge
# if you want to also use "Export to PNG" for charts
conda install -c plotly python-kaleido
conda install -c conda-forge dtale
conda install -c anaconda sqlalchemy
conda install -c anaconda tabpy-server
conda install -c conda-forge dtale
--
#setup your virtual env for query redshift with python
$ conda create --name sql pip jupyter pandas sqlalchemy psycopg2 plotly flask
$ conda activate sql
$ conda install -c conda-forge apscheduler 
$ jupyter notebook
-----


# =============================================================================
# #package install in cmd prompt
# =============================================================================
http://go.microsoft.com/fwlink/?LinkId=691126&fixForIE=.exe. # install visualc++ build tools
python -m pip install -U pip # upgrade pip
pip list

https://github.com/man-group/dtale
pip install dtale # excel like functions working directly on df
pip install tabpy
pip install pandas
pip3 install plotly==4.12.0
pip install sqlalchemy
pip install PyMySQL
pip install pyodbc
pip install ctrl4ai
pip install matplotlib
pip install pyspark
pip install scikit-learn
pip install psycopg2 #--Redshift
pip install sqlalchemy-redshift
pip install streamlit
-----------------------------------
#git hub clone repository
git clone https://github.com/qlik-demo-team/qlik-engine-tutorial my-qlik-app
cd my-qlik-app
npm install
npm run dev
----------------
# virtual environment venv
in cmd prompt
pip install virtualenv
virtualenv venv
source venv/bin/activate # not working for windows
pip install selenium

---------
mkdir pythonprojectfolder
cd pythonprojectfolder
python3 -m venv./venv
./venv/bin/activate # activate the venv
pip3.list
deactivate # go back to global environment

-----
#convert notebook to py files
> pip install ipython
> pip install nbconvert
Convert single file
> ipython nbconvert — to script abc.ipynb
You can have abc.py
Convert multi files
> ipython nbconvert — to script abc.ipynb def.ipynb
abc.py, def.py
> ipython nbconvert — to script *.ipynb
abc.py, def.py
All done!

----------
#tips trics
# count words from a file/paragraph
from collections import Counter
def word(fname):
        with open(fname) as f:
                return Counter(f.read().split())
print(word_count("text.txt"))

import csv
my_dict = word_count("text.txt")
with open('test.csv', 'w') as f:
    for key in my_dict.keys():
        f.write("%s,%s\n"%(key,my_dict[key]))
        
        
#list comprehence
data = dict
list =[fruit ['name'] for fruit in fruits if fruit[name][0] == 'a'] # fruit name starts with a 

#lambda
add = lambda x,y:x+y
add(2,3)
----
morethanone=filter(lambda x:x>1,[1,2,3]) # lambda filter 
print(list(morethanone))
print(morethanone)
evens = [x for x in range(20) if x%2 == 0 ]
print(evens)
-----
numbers = [1,2,3,4,5]
def squares(n):
    return n*n
def even(n):
    return n%2 ==0

squares = list(map(squares,numbers)) # map function
squares     
squares= list(map(lambda x: x**2,numbers))
squares
evens = list(filter(even,numbers)) #filter function
evens
evens = list(filter(lambda x: x%2 ==0,numbers))
evens
from functools import reduce
product = reduce(lambda x,y: x*y,numbers)
print(product)
-------

# Remove punctuation from sting
# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

my_str = "Hello!!!, he said ---and went."

# To take input from the user
# my_str = input("Enter a string: ")

# remove punctuation from the string
no_punct = ""
for char in my_str:
   if char not in punctuations:
       no_punct = no_punct + char

# display the unpunctuated string
print(no_punct)
----------usage
#sort words in alphabet order
# Program to sort alphabetically the words form a string provided by the user

my_str = "Hello this Is an Example With cased letters"
type(my_str)
# To take input from the user
#my_str = input("Enter a string: ")

# breakdown the string into a list of words
words = my_str.split()
type(words)
# sort the list
words.sort()

# display the sorted words

print("The sorted words are:")
for word in words:
   print(word)
type(word)
-------
# Program to perform different set operations like in mathematics

# define three sets
E = {0, 2, 4, 6, 8};
N = {1, 2, 3, 4, 5};

# set union
print("Union of E and N is",E | N)

# set intersection
print("Intersection of E and N is",E & N)

# set difference
print("Difference of E and N is",E - N)

# set symmetric difference
print("Symmetric difference of E and N is",E ^ N)
---------
# Control loop
#Number is positive /Negative
num = float(input("Enter a number: "))
if num > 0:
   print("Positive number")
elif num == 0:
   print("Zero")
else:
   print("Negative number")
-------
#Find largest among 3
# Python program to find the largest number among the three input numbers

# change the values of num1, num2 and num3
# for a different result
num1 = 10
num2 = 14
num3 = 12

# uncomment following lines to take three numbers from user
#num1 = float(input("Enter first number: "))
#num2 = float(input("Enter second number: "))
#num3 = float(input("Enter third number: "))

if (num1 >= num2) and (num1 >= num3):
   largest = num1
elif (num2 >= num1) and (num2 >= num3):
   largest = num2
else:
   largest = num3

print("The largest number is", largest)
------
