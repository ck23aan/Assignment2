#required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def data_country_col(filename):
    '''
    load the dataframe and get the country and year features and return two datframes.
    '''
    data_y = pd.read_excel(filename,engine="openpyxl")
    data_t = pd.melt(data_y, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                    var_name='Year', value_name='Value')
    data_c = data_t.pivot_table(index=['Year', 'Country Code', 'Indicator Name', 'Indicator Code'], columns='Country Name', values='Value').reset_index()
    data_c = data_c.drop_duplicates().reset_index()
    return data_y,data_c


data_y,data_c = data_country_col('world_bank_climate.xlsx')


def year_data(data,a,b,s):
    '''
    get year data feartures
    '''
    data_sample = data.copy()
    years_needed=[i for i in range(a,b,s)]
    col_need=['Country Name','Indicator Name']
    col_need.extend(years_needed)
    data_sample =  data_sample[col_need]
    data_sample = data_sample.dropna(axis=0, how="any") 
    return data_sample


data_year_filt = year_data(data_y,1990,2020,4)


countries_filt = data_year_filt['Country Name'].value_counts().index.tolist()[20:30]


def colum_specific_value(data,column,values):
    data_t= data.copy()
    data_req = data_t[data_t[column].isin(values)].reset_index(drop=True)
    return data_req


data_year_sample_country  = colum_specific_value(data_year_filt,'Country Name',countries_filt)


country_dict = dict()
for i in range(data_year_sample_country.shape[0]):
    if data_year_sample_country['Country Name'][i] not in country_dict.keys():
        country_dict[data_year_sample_country['Country Name'][i]]=[data_year_sample_country['Indicator Name'][i]]
    else:
        country_dict[data_year_sample_country['Country Name'][i]].append(data_year_sample_country['Indicator Name'][i])
    
    

for k,v in country_dict.items():
    country_dict[k] = set(v)



inter = country_dict['Brazil']
for v in country_dict.values():
    inter = inter.intersection(v)


print(data_year_sample_country.describe())


df_year_for= colum_specific_value(data_year_sample_country,'Indicator Name',['Forest area (% of land area)'])

print(df_year_for.describe())



def country_bar_plot(data,indicator_variable):
    sample_df = data.copy()
    sample_df.set_index('Country Name', inplace=True)
    numeric_columns = sample_df.columns[sample_df.dtypes == 'float64']
    sample_df = sample_df[numeric_columns]
    plt.figure(figsize=(50, 50))
    sample_df.plot(kind='bar')
    plt.title(indicator_variable)
    plt.xlabel('Country Name')    
    plt.legend(title='Year', bbox_to_anchor=(1.10, 1), loc='upper left')
    plt.show()


country_bar_plot(df_year_for,'Forest area (% of land area)')



df_year_met= colum_specific_value(data_year_sample_country,'Indicator Name',['Methane emissions (kt of CO2 equivalent)'])


print(df_year_met.describe())


country_bar_plot(df_year_met,'Methane emissions (kt of CO2 equivalent)')


df_year_brazil= colum_specific_value(data_year_sample_country,'Country Name',['Brazil'])


def data_indicator(data):
    df_req=data.copy()
    # Melt the DataFrame
    df_req = df_req.melt(id_vars='Indicator Name', var_name='Year', value_name='Value')

    # Pivot the DataFrame
    df_req = df_req.pivot(index='Year', columns='Indicator Name', values='Value')

    # Reset index
    df_req.reset_index(inplace=True)
    df_req = df_req.apply(pd.to_numeric, errors='coerce')
    del df_req['Year']
    df_req = df_req.rename_axis(None, axis=1)
    return df_req

    

data_heat_map_brazil= data_indicator(df_year_brazil)



features_need = ['Forest area (% of land area)',
 'Methane emissions (kt of CO2 equivalent)',
 'Mortality rate, under-5 (per 1,000 live births)',
 'CO2 emissions (metric tons per capita)',
 'Arable land (% of land area)',
 'Urban population growth (annual %)']




data_heat_map_brazil_map = data_heat_map_brazil[features_need]


print(data_heat_map_brazil_map.corr())



sns.heatmap(data_heat_map_brazil_map.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')



df_mort= colum_specific_value(data_year_sample_country,'Indicator Name',['Mortality rate, under-5 (per 1,000 live births)'])



print(df_mort.describe())

df_urban= colum_specific_value(data_year_sample_country,'Indicator Name',['Mortality rate, under-5 (per 1,000 live births)'])




def time_plotting_feature(data,feature_label):
    data_verify = data.copy()
    data_verify.set_index('Country Name', inplace=True)
    num_col = data_verify.columns[data_verify.dtypes == 'float64']
    data_verify = data_verify[num_col]

    plt.figure(figsize=(12, 6))
    for count in data_verify.index:
        plt.plot(data_verify.columns, data_verify.loc[count], label=count, linestyle='dashed', marker='o')

    plt.title(feature_label)
    plt.xlabel('Year')
    plt.legend(title='Country', bbox_to_anchor=(1.15, 1), loc='upper left')

    plt.show()


time_plotting_feature(df_mort,'Mortality rate, under-5 (per 1,000 live births)')


df_arb= colum_specific_value(data_year_sample_country,'Indicator Name',['Arable land (% of land area)'])


print(df_arb.describe())



time_plotting_feature(df_arb,'Arable land (% of land area)')



df_year_col_nig = colum_specific_value(data_year_sample_country,'Country Name',['Nigeria'])
df_year_col_nig = data_indicator(df_year_col_nig)
df_year_col_nig = df_year_col_nig[features_need]
sns.heatmap(df_year_col_nig.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')


df_year_col_maur= colum_specific_value(data_year_sample_country,'Country Name',['Mauritius'])
df_year_col_maur = data_indicator(df_year_col_maur)
df_year_col_maur = df_year_col_maur[features_need]
plt.figure()
sns.heatmap(df_year_col_maur.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')


