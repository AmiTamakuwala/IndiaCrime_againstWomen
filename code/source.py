import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# import data:
crime_data = pd.read_csv("D:\\Crime_against_Women\\42_District_wise_crimes_committed_against_women_2001_2012.csv")
print(crime_data)

# checking unique values of "STATE/UT" column and length of the column:

print(crime_data["STATE/UT"].unique())
print(len(crime_data["STATE/UT"].unique()))

# checking null values:

print(crime_data.isnull().sum())

crime_in_years_df = crime_data.groupby('Year').sum()
crime_in_years_df.reset_index(inplace=True)
print(crime_in_years_df)

# total number of crime between 2001 to 2012.

crime_in_years_df['Total Number of Cases'] = crime_in_years_df.sum(axis=1)
print('Total no of crime cases against women between 2001 to 2012: ',
crime_in_years_df['Total Number of Cases'].sum())

# crime_in_years_df.plot (x='Year', y='Rape', title = "Rape Cases in India between 2001 to 2012", 
#                         kind = 'bar', color = "Red")

# crime_in_years_df.plot (x='Year', y='Kidnapping and Abduction',
#                         title = "Kidnapping and Abduction cases in India between 2001 to 2012", 
#                         kind = 'bar', color = "Yellow")

# crime_in_years_df.plot (x='Year', y='Dowry Deaths',
#                         title = "Dowry Deaths cases in India between 2001 to 2012", 
#                         kind = 'bar', color = "Blue")

# crime_in_years_df.plot (x='Year', y='Assault on women with intent to outrage her modesty',
#                         title = "Assault to her modesty in India between 2001 to 2012", 
#                         kind = 'bar', color = "gray")

# crime_in_years_df.plot (x='Year', y='Insult to modesty of Women', 
#                         title = "Insult to modesty of Women in India between 2001 to 2012", 
#                         kind = 'bar', color = "skyblue")

# crime_in_years_df.plot (x='Year', y='Cruelty by Husband or his Relatives', 
#                         title = "Cruelty agaist women crime in India between 2001 to 2012", 
#                         kind = 'bar', color = "brown")

# crime_in_years_df.plot (x='Year', y='Importation of Girls', 
#                         title = "Cases of Importation of Girls between 2001 to 2012", 
#                         kind = 'bar', color = "indianred")
# plt.show()

# creating a line chart for total number of cases in year 2001 to 2012.

plt.plot(crime_in_years_df['Year'], crime_in_years_df['Total Number of Cases'], color='red', marker='*')
plt.title('total crimes against women in India')
plt.xlabel('Years', fontsize = 12)
plt.ylabel("Count of Crime Cases", fontsize=12)
plt.grid(True)
plt.show()

# Percentage of each crime
# in between 2001-2012 through line graph and pie-chart:

print(crime_in_years_df.head())

print(crime_in_years_df.columns)

# Rate of change of different crime over time:

plt.figure(figsize=(12,10))
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Rape'], color='red', marker = '.')
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Kidnapping and Abduction'], color='blue', marker = 'o')
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Dowry Deaths'], color='green', marker = 'o')
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Assault on women with intent to outrage her modesty'], color='cyan', marker = 'x')
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Insult to modesty of Women'], color='orange', marker = '4')
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Cruelty by Husband or his Relatives'], color='brown', marker = '3')
plt.plot(crime_in_years_df['Year'], crime_in_years_df['Importation of Girls'], color='skyblue', marker = 's')
plt.title("yearly crime cases against Women")
plt.xlabel("Years")
plt.ylabel("Sprade Of Cases")
plt.legend(['Rape', 'Kidnapping and Abduction', 'Dowry Deaths',
       'Assault on women with intent to outrage her modesty',
       'Insult to modesty of Women', 'Cruelty by Husband or his Relatives',
       'Importation of Girls'], loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()

# Which was highest reported crime and which one was least?
# which crime is highest and lowest reported for that we we remove last column "Total Number Of Cases".

# use "drop" for remove "Total Number Of Cases" column:

crime_df = crime_in_years_df.drop("Total Number of Cases", axis=1)
crime_df.set_index("Year", inplace = True)
print(crime_df)

# let's sort the values of the total count of the crime. 

Total_crime_df = pd.DataFrame(crime_df.sum(axis=0), columns = ['Count']).sort_values(by='Count', ascending=False)
print(Total_crime_df)

# Percentage of each crime in between 2001-2012 Pir-chart:¶

dataframe = pd.DataFrame({'Crime':["Cruelty by Husband or his Relatives", 
                                "Assault on women with intent to outrage her modesty",
                                "Kidnapping and Abduction",
                                "Rape", "Insult to modesty of Women",
                                "Dowry Deaths", "Importation of Girls"],
                         'Count':[1750402, 906310, 527814, 
                                  478274, 248108, 182404, 1784]})

# Defining colors for the pie chart:
colors = ["pink", "red", "steelblue", "red", "orange",
         "indigo", "violet"]

explode = (0.0, 0.1, 0.0, 0.30, 0.0, 0.0, 0.0)

dataframe.groupby(['Crime']).sum().plot(
kind = 'pie', y = 'Count',
autopct='%1.0f%%',
colors = colors, explode= explode, figsize=(15,10))
plt.show()

##################       State/UT wise Analysis:        ####################
# as per above text table there are no null values in our dataset. 
# So, we don't have anything else to clean in our data.

# merging the State/UT column:
# and also we will drop "Year" column:

State_UT_crime_df = crime_data.groupby('STATE/UT').sum()
State_UT_crime_df.reset_index(inplace=False)
print(State_UT_crime_df)

# drop the "Year" column:

State_UT_crime_df = State_UT_crime_df.drop("Year", axis=1)
print(State_UT_crime_df)

# checking column and rows:

print(State_UT_crime_df.shape)

# Top 10 State/Union Territories with "Highest" number of crime Cases:

print(pd.DataFrame(State_UT_crime_df.sum(axis=1), columns=['Total Cases']).sort_values(by='Total Cases', 
    ascending=False).head(10))

# Top 10 State/Union Territories with "Lowest" number of crime Cases:

print(pd.DataFrame(State_UT_crime_df.sum(axis=1), columns=["Total Cases"]).sort_values(by='Total Cases').head(10))

# Out Of 7 types of Crimes, which State/UT haing the "max" 
# cases of each type crime against women in India between 2001 to 2012?

print(pd.DataFrame(State_UT_crime_df.idxmax(), columns=["STATE/UT"]))

# Out Of 7 types of Crimes, which State/UT haing the "min" 
# cases of each type crime against women in India between 2001 to 2012?

print(pd.DataFrame(State_UT_crime_df.idxmin(), columns=["STATE/UT"]))

# Function for Specific Satte/UT Analysis:
# we are making function for, which State/UT crime cases against women you want to analyze data
# (between 2001 to 2012- from the data we have). 
# You just need to type the name of the State/UT:

colors = ["pink", "red", "skyblue", "yellow", "orange",
         "violet", "purple"]

crime_type = ["Rape","Kidnapping and Abduction", "Dowry Deaths",
                "Assault on women with intent to outrage her modesty",
                "Insult to modesty of Women",
                "Cruelty by Husband or his Relatives",
                "Importation of Girls"]

explode = [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.30]

labels = list(crime_type)
def which_state_want_to_analyze(State_name):
    '''
    this function will give you the data of crime agaist women 
    from which satate/UT you want!
    '''
    try:
        fig = plt.figure(figsize=(15,12))
        plt.pie(State_UT_crime_df.loc[State_name],
                labels = labels,
                explode = explode, colors = colors, 
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.show()
    except KeyError:
        print("You Entered Wrong STATE/UT Name.")
        
State_name = input("Please enter Name of STATE/UT: ").upper()
print(which_state_want_to_analyze(State_name))

print(State_UT_crime_df.info())

















































































































