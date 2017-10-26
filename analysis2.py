import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
np.random.seed(5)
pd.set_option('chained_assignment',None)

df3 = pd.read_csv("./dataset/nas-pupil-marks.csv")
dfx = df3[['STUID', 'State']]
nd = df3[['Gender', 'Age', 'Category',
       'Same language', 'Siblings', 'Handicap', 'Father edu', 'Mother edu',
       'Father occupation', 'Mother occupation', 'Below poverty',
       'Use calculator', 'Use computer', 'Use Internet', 'Use dictionary',
       'Read other books', '# Books', 'Distance', 'Computer use',
       'Library use', 'Like school', 'Subjects', 'Give Lang HW',
       'Give Math HW', 'Give Scie HW', 'Give SoSc HW', 'Correct Lang HW',
       'Correct Math HW', 'Correct Scie HW', 'Correct SocS HW',
       'Help in Study', 'Private tuition', 'English is difficult',
       'Read English', 'Dictionary to learn', 'Answer English WB',
       'Answer English aloud', 'Maths is difficult', 'Solve Maths',
       'Solve Maths in groups', 'Draw geometry', 'Explain answers',
       'SocSci is difficult', 'Historical excursions', 'Participate in SocSci',
       'Small groups in SocSci', 'Express SocSci views',
       'Science is difficult', 'Observe experiments', 'Conduct experiments',
       'Solve science problems', 'Express science views', 'Watch TV',
       'Read magazine', 'Read a book', 'Play games', 'Help in household',
       'Maths %', 'Reading %', 'Science %', 'Social %']]

nd['Use computer'] = nd['Use computer'].map({"Yes":1,"No":0})
nd['Subjects'] = nd['Subjects'].map({'L':1, 'S':2, 'O':3, 'M':4, '0':0})
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(nd)
tx = imp.transform(nd) 

dc=['Gender', 'Age', 'Category',
       'Same language', 'Siblings', 'Handicap', 'Father edu', 'Mother edu',
       'Father occupation', 'Mother occupation', 'Below poverty',
       'Use calculator', 'Use computer', 'Use Internet', 'Use dictionary',
       'Read other books', '# Books', 'Distance', 'Computer use',
       'Library use', 'Like school', 'Subjects', 'Give Lang HW',
       'Give Math HW', 'Give Scie HW', 'Give SoSc HW', 'Correct Lang HW',
       'Correct Math HW', 'Correct Scie HW', 'Correct SocS HW',
       'Help in Study', 'Private tuition', 'English is difficult',
       'Read English', 'Dictionary to learn', 'Answer English WB',
       'Answer English aloud', 'Maths is difficult', 'Solve Maths',
       'Solve Maths in groups', 'Draw geometry', 'Explain answers',
       'SocSci is difficult', 'Historical excursions', 'Participate in SocSci',
       'Small groups in SocSci', 'Express SocSci views',
       'Science is difficult', 'Observe experiments', 'Conduct experiments',
       'Solve science problems', 'Express science views', 'Watch TV',
       'Read magazine', 'Read a book', 'Play games', 'Help in household',
       'Maths %', 'Reading %', 'Science %', 'Social %']
data = pd.DataFrame(tx,columns=dc)

math    = np.array(data["Maths %"]).astype("float")
reading = np.array(data["Reading %"]).astype("float")
Science = np.array(data["Science %"]).astype("float")
Social  = np.array(data["Social %"]).astype("float")
performance = (math+reading+Science+Social)

bestPerformance = np.max(performance)
poorPerformance = np.min(performance)
avgPerformance  = np.average(performance)

Threshold = bestPerformance-avgPerformance
super_threshold_indices = performance > Threshold
a = np.copy(performance)
a[super_threshold_indices] = 1
a[~super_threshold_indices]= 0

performance = pd.DataFrame(performance,columns=["performance"])
Stclass     = pd.DataFrame(a,columns=["Stclass"])
state = pd.DataFrame(np.array(df3['State']),columns=["state"])

modDf = pd.concat([state,data,performance,Stclass],axis=1)
sNm=modDf.state.unique()

def girlVsboy(modDf,state_name):
    state_           = modDf[modDf["state"]==state_name]
    state_boy        = state_[state_['Gender']==1].reset_index(drop=True)
    state_girl       = state_[state_['Gender']==2].reset_index(drop=True)
    performance_boy  = state_boy['Stclass']
    performance_girl = state_girl['Stclass']
    return (performance_girl,performance_boy)

Total_nof_bestPrfmGr=[]
Total_nof_bestPrfmBy=[]
Total_nof_poorPrfmGr=[]
Total_nof_poorPrfmBy=[]

for i in sNm:
    tmp   = girlVsboy(modDf,i)
    bestG = np.sum(tmp[0])
    bestB = np.sum(tmp[1])
    poorG = len(tmp[0])-bestG
    poorB = len(tmp[1])-bestB
    Total_nof_bestPrfmGr.append(bestG)
    Total_nof_bestPrfmBy.append(bestB)
    Total_nof_poorPrfmGr.append(poorG)
    Total_nof_poorPrfmBy.append(poorB)

import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Bar(
    x = sNm,
    y = Total_nof_bestPrfmGr,
    name='Girl'
)
trace2 = go.Bar(
    x = sNm,
    y = Total_nof_bestPrfmBy,
    name='Boy'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Best Performance of Girls and Boys over states',
    xaxis=dict(
        title='States',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No of Student',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Best_Performance_of_Girls_and_Boys_over_states')
#plt.savefig("f1.png")

trace1 = go.Bar(
    x = sNm,
    y = Total_nof_poorPrfmGr,
    name='Girl'
)
trace2 = go.Bar(
    x = sNm,
    y = Total_nof_poorPrfmBy,
    name='Boy'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    title='Poor Performance of Girls and Boys over states',
    xaxis=dict(
        title='States',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='No of Student',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Poor_Performance_of_Girls_and_Boys_over_states')
#plt.savefig("f2.png")