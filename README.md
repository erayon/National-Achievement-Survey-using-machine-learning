# National-Achievement-Survey-using-machine-learning
National Council of Education Research and Training conducts yearly National Achievement Survey. provided the data of Class VIII students from 2014.


# Prequisition

```
pip install pandas
pip install numpy
pip install sklearn
pip install plotly

```

## Inside the Dataset
1. nas-columns.csv : consist of details names of columns and their type
2. nas-labels.csv : details of each colum

```
        Column	         Name	      Level       Rename
87	Subjects	Language        L	  Language
88	Subjects	Mathematics	M	  Mathematics
89	Subjects	None	        0	  None
90	Subjects	Science	        S	  Science
91	Subjects	Social Science  O	  Social Science

```
3. nas-pupil-marks.csv : actual servey dataset consist of feaures of 

```
['STUID', 'State', 'District', 'Gender', 'Age', 'Category',
'Same language', 'Siblings', 'Handicap', 'Father edu', 'Mother edu',
'Father occupation', 'Mother occupation', 'Below poverty',
'Use calculator', 'Use computer', 'Use Internet', 'Use dictionary',
'Read other books', '# Books', 'Distance', 'Computer use',
'Library use', 'Like school', 'Subjects', 'Give Lang HW',
'Give Math HW', 'Give Scie HW', 'Give SoSc HW', 'Correct Lang HW'
,'Correct Math HW', 'Correct Scie HW', 'Correct SocS HW',
,'Help in Study', 'Private tuition', 'English is difficult'
,'Read English', 'Dictionary to learn', 'Answer English WB'
,'Answer English aloud', 'Maths is difficult', 'Solve Maths'   
'Solve Maths in groups', 'Draw geometry', 'Explain answers'
'SocSci is difficult', 'Historical excursions', 'Participate in SocSci',
'Small groups in SocSci', 'Express SocSci views',
'Science is difficult', 'Observe experiments', 'Conduct experiments'
,'Solve science problems', 'Express science views', 'Watch TV',
'Read magazine', 'Read a book', 'Play games', 'Help in household',
'Maths %', 'Reading %', 'Science %', 'Social %']
```

# Three Question to solve using this survey Dataset
## 1. What influences students performance the most? (analysis1.ipynb)
Based on the feature or attributes which attributes influance student most on their overall performance. here I use only 'nas-pupil-marks.csv' dataset.
1. Remove STUDID, State and District 
2. Change Categorical data to numerical from columns 'Use Computer' and 'Subjects' by maping method in pandas
```
nd['Use computer'] = nd['Use computer'].map({"Yes":1,"No":0})
nd['Subjects'] = nd['Subjects'].map({'L':1, 'S':2, 'O':3, 'M':4, '0':0})
```
3. Preprossing the Dataset, remove nan value using sklearn.preprocessing Imputer
4. Now take math, reading, Science and Social and find the Performance and find the Thresholding value by which we can classify the student.
If a student performance greater than equal the threshold then it consider as a best (1) student and if performance is less than the threshold then its consider as poor (0) student. Create a new column name lable based on that.

```
math    = np.array(data["Maths %"]).astype("float")
reading = np.array(data["Reading %"]).astype("float")
Science = np.array(data["Science %"]).astype("float")
Social  = np.array(data["Social %"]).astype("float")

performance = (math+reading+Science+Social)

bestPerformance = np.max(performance)
poorPerformance = np.min(performance)
avgPerformance  = np.average(performance)

Threshold = bestPerformance-avgPerformance

```
5. Split the Dataset as a lable and Xdata such a way that label consist of 0 and 1 means best and poor. 
Xdata consist of remaning attributes.
6. Find the feature importance using ExtraTreesClassfire in sklearn.ensemble.
after fit the model its gives the feture_importance
![Alt text](influence.png?raw=true "influence") 



## 2. How do boys and girls perform across states? (analysis2.ipynb)
Based on the feature or attributes performance of boys and girls student state wise.

1. Remove STUDID, District

### Step 2. and Step 3. and Step 4. are same as above
5. Create a method which takes table and statename as an argument and return performance of girls and boys in that state, then find out best girl, best boy, poor girl, poor boy performance over states. Using as Thresholding.
![Alt text](im1.png?raw=true "states") 

## 3. Do students from South Indian states really excel at Math and Science? (analysis3.ipynb)
1. Remove STUDID, District
### Step 2. and Step 3. and Step 4. are same as above
5. Create a array southInd = ['AP','GA','KA','KL','PY','TN'] 
```
'AP' : Andhra Pradesh
'GA' : Goa
'KA' : Karnataka
'KL' : Kerala
'PY' : Pondicherry
'TN' : Tamil Nadu
```
6. And same as above create a Create a method which takes table and southInd array as an argument and return performance of girls and boys in that state, then find out best girl, best boy, poor girl, poor boy performance over states. Using as Thresholding.
![Alt text](im2.png?raw=true "southinda") 


















