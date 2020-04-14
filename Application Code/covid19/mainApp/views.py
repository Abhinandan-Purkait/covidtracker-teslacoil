from plotly.offline import plot, iplot, init_notebook_mode
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import urlopen
from datetime import timedelta
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.offline import plot
from django.shortcuts import render
import requests
from bs4 import BeautifulSoup
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import cv2
import keras
import warnings
from .models import Images

# MKIT STARTS
def load_models():
    model1=keras.models.load_model("models/first_level_hierarchy.h5")
    model2=keras.models.load_model("models/second_level_hierarchy.h5")
    return model1,model2

def test_pneumonia(image,model1,model2):
    logs=["Covid 19","Bacterial Pneumonia","Viral Pneumonia","Negative"]
    result=dict()
    image=(np.array([cv2.resize(image,(150,150))]).reshape(1,150,150,3)).astype('float32')/255
    base=model1.predict(image)
    indx=np.argmax(base)
    if indx==1:
        derived=model2.predict(image)
        indx_der=np.argmax(derived)
        result['Pneumonia']=[logs[indx_der],derived[0][indx_der]*100]
    elif indx==0:
        result['Pneumonia']=[logs[3],base[0][indx]*100]
    return(result)

# MKIT ENDS

# statistics start
plots = []

data = pd.read_csv(
    "datasets/covid19GlobalForecastingweek1/train.csv", parse_dates=['Date'])
cleaned_data = pd.read_csv(
    "datasets/covid19cleancompletedataset/covid_19_clean_complete.csv", parse_dates=['Date'])
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']
cleaned_data['Active'] = cleaned_data['Confirmed'] - \
    cleaned_data['Deaths'] - cleaned_data['Recovered']
cleaned_data['Country/Region'] = cleaned_data['Country/Region'].replace(
    'Mainland China', 'China')
cleaned_data[['Province/State']
             ] = cleaned_data[['Province/State']].fillna('')
cleaned_data[cases] = cleaned_data[cases].fillna(0)
cleaned_data.rename(columns={'Date': 'date'}, inplace=True)
data = cleaned_data

grouped = data.groupby(
    'date')['date', 'Confirmed', 'Deaths', 'Active'].sum().reset_index()
grouped = data.groupby(
    'date')['date', 'Confirmed', 'Deaths', 'Active'].sum().reset_index()

fig1 = px.line(grouped, x="date", y="Deaths",
               title="Worldwide Death Cases Over Time")

grouped_india = data[data['Country/Region'] == "India"].reset_index()
grouped_india_date = grouped_india.groupby(
    'date')['date', 'Confirmed', 'Deaths'].sum().reset_index()
plot_titles = ['India']

fig2 = px.line(grouped_india_date, x="date", y="Confirmed",
               title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", color_discrete_sequence=['#F61067'], height=500)

data['Province/State'] = data['Province/State'].fillna('')
temp = data[[col for col in data.columns if col != 'Province/State']]
latest = temp[temp['date'] == max(temp['date'])].reset_index()
latest_grouped = latest.groupby(
    'Country/Region')['Confirmed', 'Deaths'].sum().reset_index()

fig3 = px.bar(latest_grouped.sort_values('Confirmed', ascending=False)[
    :40][::-1], x='Confirmed', y='Country/Region', title='Confirmed Cases Worldwide', text='Confirmed', height=1000, orientation='h')

fig4 = px.bar(latest_grouped.sort_values('Deaths', ascending=False)[:30][::-1], x='Deaths', y='Country/Region', color_discrete_sequence=[
    '#84DCC6'], title='Deaths Cases Worldwide', text='Deaths', height=1000, orientation='h')

temp = cleaned_data.groupby(
    'date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="date", value_vars=[
    'Recovered', 'Deaths', 'Active'], var_name='case', value_name='count')
temp['case'].value_counts()
pio.templates.default = "plotly_dark"

fig5 = px.line(temp, x="date", y="count", color='case',
               title='Cases over time: Line Plot', color_discrete_sequence=['cyan', 'red', 'orange'])

fig6 = px.area(temp, x="date", y="count", color='case',
               title='Cases over time: Area Plot', color_discrete_sequence=['cyan', 'red', 'orange'])

formated_gdf = data.groupby(
    ['date', 'Country/Region'])['Confirmed', 'Deaths'].max()

formated_gdf = data.groupby(
    ['date', 'Country/Region'])['Confirmed', 'Deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig8 = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names',
                      color="Confirmed", size='size', hover_name="Country/Region",
                      range_color=[0, 1500],
                      projection="natural earth", animation_frame="date",
                      title='COVID-19: Spread Over Time', color_continuous_scale="portland")

formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)
pio.templates.default = "plotly_dark"
fig7 = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', color="Deaths", size='size', hover_name="Country/Region",
                      range_color=[0, 100],  projection="natural earth", animation_frame="date",  title='COVID-19: Deaths Over Time', color_continuous_scale="peach")

# statistics end


# symptoms start

symptoms = {'symptom': ['Fever',
                        'Dry cough',
                        'Fatigue',
                        'Sputum production',
                        'Shortness of breath',
                        'Muscle pain',
                        'Sore throat',
                        'Headache',
                        'Chills',
                        'Nausea or vomiting',
                        'Nasal congestion',
                        'Diarrhoea',
                        'Haemoptysis',
                        'Conjunctival congestion'], 'percentage': [87.9, 67.7, 38.1, 33.4, 18.6, 14.8, 13.9, 13.6, 11.4, 5.0, 4.8, 3.7, 0.9, 0.8]}

symptoms = pd.DataFrame(data=symptoms, index=range(14))

symptom_graph = px.bar(symptoms[['symptom', 'percentage']].sort_values('percentage', ascending=False),
                       y="percentage", x="symptom", color='symptom',
                       log_y=True, template='ggplot2', title='Symptom of  Coronavirus')
symptom_div = plot(symptom_graph, output_type='div', include_plotlyjs=False, show_link=False,
                   link_text="", image_width=500, config={"displaylogo": False})
plots.append(symptom_div)

# symptoms end

# india starts


cnf = '#393e46'  # confirmed - grey
dth = '#ff2e63'  # death - red
rec = '#21bf73'  # recovered - cyan
act = '#fe9801'  # active case - yellow

register_matplotlib_converters()
pio.templates.default = "plotly"

# importing datasets
df = pd.read_csv('datasets/complete.csv', parse_dates=['Date'])
df['Name of State / UT'] = df['Name of State / UT'].str.replace(
    'Union Territory of ', '')
df = df[['Date', 'Name of State / UT', 'Latitude', 'Longitude',
         'Total Confirmed cases', 'Death', 'Cured/Discharged/Migrated']]
df.columns = ['Date', 'State/UT', 'Latitude',
              'Longitude', 'Confirmed', 'Deaths', 'Cured']

for i in ['Confirmed', 'Deaths', 'Cured']:
    df[i] = df[i].astype('int')

df['Active'] = df['Confirmed'] - df['Deaths'] - df['Cured']
df['Mortality rate'] = df['Deaths']/df['Confirmed']
df['Recovery rate'] = df['Cured']/df['Confirmed']

df = df[['Date', 'State/UT', 'Latitude', 'Longitude', 'Confirmed',
         'Active', 'Deaths', 'Mortality rate', 'Cured', 'Recovery rate']]
latest = df[df['Date'] == max(df['Date'])]

# days
latest_day = max(df['Date'])
day_before = latest_day - timedelta(days=1)

# state and total cases
latest_day_df = df[df['Date'] == latest_day].set_index('State/UT')
day_before_df = df[df['Date'] == day_before].set_index('State/UT')

temp = pd.merge(left=latest_day_df, right=day_before_df,
                on='State/UT', suffixes=('_lat', '_bfr'), how='outer')
latest_day_df['New cases'] = temp['Confirmed_lat'] - temp['Confirmed_bfr']
latest = latest_day_df.reset_index()
latest.fillna(1, inplace=True)

temp = latest.sort_values('Confirmed', ascending=False)
states = temp['State/UT']

fig = go.Figure(data=[
    go.Bar(name='Confirmed', x=states, y=temp['Confirmed']),
    go.Bar(name='Recovered', x=states, y=temp['Cured']),
    go.Bar(name='Deaths', x=states, y=temp['Deaths'])
])
# Change the bar mode
fig9 = fig.update_layout(barmode='group')

fig10 = px.scatter(latest[latest['Confirmed'] > 10], x='Confirmed', y='Deaths', color='State/UT', size='Confirmed',
                   text='State/UT', log_x=True, title='Confirmed vs Death')

figs = [fig1, fig2, fig3, fig4, fig5, fig6,
        fig7, fig8, fig9, fig10]

for x in figs:
    plot_div = plot(x, output_type='div', include_plotlyjs=False, show_link=False,
                    link_text="", image_width=500, config={"displaylogo": False})
    plots.append(plot_div)


def coronaCount(url):
    counts = []
    r = requests.get(url)
    c = r.content
    soup = BeautifulSoup(c, 'html.parser')
    qs = soup.find_all('div', {'class': 'maincounter-number'})
    for x in qs:
        counts.append(int(x.text.replace(',', '')))
    qs1 = soup.find('div', {'class': 'number-table-main'})
    counts.append(int(qs1.text.replace(',', '')))
    return counts


def index(request):
    return render(request, 'index.html')


def statistics(request):
    global plot
    return render(request, "statistics.html", context={'plot_div1': plots[1], 'plot_div2': plots[2], 'plot_div3': plots[3], 'plot_div4': plots[4], 'plot_div5': plots[5], 'plot_div6': plots[6], 'plot_div9': plots[9], 'plot_div10': plots[10]})


def prevention(request):
    return render(request, 'prevention.html')


def symptoms(request):
    global plots
    return render(request, 'symptoms.html', context={'symp': plots[0]})


def faq(request):
    return render(request, 'faq.html')


def map_stats(request):
    global plots
    return render(request, 'india_map.html', context={'plot_div7': plots[7], 'plot_div8': plots[8]})


def prediction(request):
    return render(request, 'prediction.html')

def about(request):
    return render(request, 'about.html')



def vitualMedicalKit(request):
    if request.method == 'POST':
        model1,model2=load_models()
        typelis = []
        problis = []
        imgs = []
        for count,x in enumerate(request.FILES.getlist('image')):
            img = Images()
            print(x, "**")
            img.image = x
            print(x, "**")
            img.save()
            print(x, "**")
            imgs.append(str(x))
        
        for x in imgs:
            imageFile=cv2.imread("media/images/"+x)
            out = test_pneumonia(imageFile,model1,model2)
            res = out['Pneumonia']
            typelis.append(res[0])
            problis.append(res[1])
        
        abc = Images.objects.all()
        abc.delete()
        mainlist = zip(imgs, typelis, problis)
        return render(request, 'mkit.html',{'lis':mainlist,"Res":'result'})
        return render(request, 'mkit.html')
    else:
        return render(request, 'mkit.html')

def indiaAnalysis(request):
    plotsInd = []
    df_carona_in_india = pd.read_csv("datasets/Covid_19_India/covid_19_india.csv")

    #Total cases of carona in India
    df_carona_in_india['Total Cases'] = df_carona_in_india['Cured'] + df_carona_in_india['Deaths'] + df_carona_in_india['Confirmed']
    #Active cases of carona in India
    df_carona_in_india['Active Cases'] = df_carona_in_india['Total Cases'] - df_carona_in_india['Cured'] - df_carona_in_india['Deaths']
    df_carona_in_india.head()
    #Till 10th April Active Cases in India
    df1= df_carona_in_india[df_carona_in_india['Date']=='10/04/20']
    fig = px.bar(df1, x='State/UnionTerritory', y='Active Cases', color='Active Cases',barmode='group', height=600)
    fig.update_layout(title='Till 10th April Active Cases in India')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    df_carona_in_india['Date'] =pd.to_datetime(df_carona_in_india.Date,dayfirst=True)
    df_carona_in_india.head()
    #Daily Cases in India Datewise
    carona_data = df_carona_in_india.groupby(['Date'])['Total Cases'].sum().reset_index().sort_values('Total Cases',ascending = True)
    carona_data['Daily Cases'] = carona_data['Total Cases'].sub(carona_data['Total Cases'].shift())
    carona_data['Daily Cases'].iloc[0] = carona_data['Total Cases'].iloc[0]
    carona_data['Daily Cases'] = carona_data['Daily Cases'].astype(int)
    fig = px.bar(carona_data, y='Daily Cases', x='Date',hover_data =['Daily Cases'], color='Daily Cases', height=500)
    fig.update_layout(title='Daily Cases in India Datewise')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    carona_data['Corona Growth Rate'] = carona_data['Total Cases'].pct_change().mul(100).round(2)
    #Corona Growth Rate Comparison with Previous Day
    fig = px.bar(carona_data, y='Corona Growth Rate', x='Date',hover_data =['Corona Growth Rate','Total Cases'], height=500)
    fig.update_layout(title='Corona Growth Rate(in Percentage) Comparison with Previous Day')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    #Moratality Rate
    carona_data = df_carona_in_india.groupby(['Date'])['Total Cases','Active Cases','Deaths'].sum().reset_index().sort_values('Date',ascending=False)
    carona_data['Mortality Rate'] = ((carona_data['Deaths']/carona_data['Total Cases'])*100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Mortality Rate'],mode='lines+markers',name='Cases'))
    fig.update_layout(title_text='COVID-19 Mortality Rate in INDIA',plot_bgcolor='rgb(225,230,255)')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    #Total Cases in India State Datewise
    carona_data = df_carona_in_india.groupby(['Date','State/UnionTerritory','Total Cases'])['Cured','Deaths','Active Cases'].sum().reset_index().sort_values('Total Cases',ascending = False)
    fig = px.bar(carona_data, y='Total Cases', x='Date',hover_data =['State/UnionTerritory','Active Cases','Deaths','Cured'], color='Total Cases',barmode='group', height=700)
    fig.update_layout(title='Indian States with Current Total Corona Cases')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    #Pie chart visualization of states effected by caronavirus
    df_Age = pd.read_csv("datasets/Covid_19_India/AgeGroupDetails.csv")
    fig = px.pie(df_Age, values='TotalCases', names='AgeGroup')
    fig.update_layout(title='Age Group affected with COVID-19')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    #Total Cases,Active Cases,Cured,Deaths from Corona Virus in India
    carona_data = df_carona_in_india.groupby(['Date'])['Total Cases','Active Cases','Cured','Deaths'].sum().reset_index().sort_values('Date',ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Total Cases'],
                        mode='lines+markers',name='Total Cases'))
    fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Active Cases'], 
                    mode='lines+markers',name='Active Cases'))
    fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Cured'], 
                    mode='lines+markers',name='Cured'))
    fig.add_trace(go.Scatter(x=carona_data['Date'], y=carona_data['Deaths'], 
                    mode='lines+markers',name='Deaths'))
    fig.update_layout(title_text='Curve Showing Different Cases from COVID-19 in India',plot_bgcolor='rgb(225,230,255)')
    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    #ARIMA Model
    import datetime
    prediction_dates = []
    start_date = carona_data['Date'][len(carona_data) - 1]
    for i in range(100):
        date = start_date + datetime.timedelta(days=1)
        prediction_dates.append(date)
        start_date = date
    df = pd.DataFrame()
    df['Dates'] = prediction_dates
    #df['Halt_Prediction'] = fcast1
    carona_data.head()
    arima_data = carona_data.drop(['Total Cases','Cured','Deaths'],axis=1)

    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import statsmodels.api as sm
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras.layers import Dropout
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    model = ARIMA(arima_data['Active Cases'].values, order=(1, 2, 1))
    fit_model = model.fit(trend='c', full_output=True, disp=True)
    fit_model.summary()
    forcast = fit_model.forecast(steps=100)
    pred_y = forcast[0].tolist()
    pd.DataFrame(pred_y)
    df = pd.DataFrame()
    df['Dates'] = prediction_dates
    df['ARIMA_Prediction'] = pred_y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Dates'], y=df['ARIMA_Prediction'],mode='lines+markers',name='Predicted Active Cases'))
    fig.update_layout(title_text='Curve Showing Predicted Active Cases from COVID-19 in India using ARIMA Model',plot_bgcolor='rgb(225,230,255)')

    plot_div = plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="", image_width=500, config={"displaylogo": False})
    plotsInd.append(plot_div)

    return render(request, 'indiaAnalysis.html', context={'plots': plotsInd})