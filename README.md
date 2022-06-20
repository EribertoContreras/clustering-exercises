# Zillow Clustering Project

INTRODUCTION
GOALS
Identify drivers of logerror.

Use clustering algorithms to identify predictors of logerror so that we can improve predictive property value models.

use clustering algorithms and adapt them to the predictive model to beat the baseline model.

PLAN


1. Acquire Data

Takeaways:

- Data is collected from the codeup cloud database with an appropriate SQL query.
- Data is imported using an acquire.py file.
- Original dataframe consisted of 71858 rows × 69 columns.
- Null values/ missing data are very common in about 50 percent of the data.

2. Prepare

- using wrangle.py

Takeaways:

- Be fore cleaning data and dropping unnecesary columns, 71858 rows × 69 columns.
- After dropping nulls and collumns, 44679 rows × 12 columns.
- resulted in 62% row retention 17% column retention.
- we continued to split the data into train, validate, and test for exploration and modeling purposes.


- Questions to answer while exploring:

- Is there a correlation between bathroomcnt & logerror ?
- Is there a correlation between calculatedfinishedsquarefeet & logerror?
- Is there a correlation between bedroomcnt & logerror ?
- Is there a correlation between yearbuilt & logerror ?
- Is there a correlation between fips & logerror ?
- Is there a correlation between taxvaluedollarcnt & logerror ?


3a. Explore1
-answers:

- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between bathroomcnt and logerror.
- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between calculatedfinishedsquarefeet and logerror.
- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between bedroomcnt and logerror.
- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between yearbuilt and logerror.
- Based on the correlation test above fips and log error have no correlation.
- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between taxvaluedollarcnt and logerror.


3b.Explore2
-answers:

- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between bathroomcnt and logerror.
- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between calculatedfinishedsquarefeet and logerror.
- After running a corralation test we can reject the null hypothesis and prove that there is correlation between bedroomcnt and logerror.
- After running a corralation test We reject the null hypothesis because there is significant correlation with year built and logerror present.
- Based on the correlation test above fips and log error have no correlation.
- After running a corralation test We fail to reject the null hypothesis because there is no significant correlation between taxvaluedollarcnt anf logerror present.


3c.Explore3
-answers:

- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between bathroomcnt and logerror
- After running a corralation test we can reject the null hypothesis and prove that there is a correlation between calculatedfinishedsquarefeet and logerror.
- After running a corralation test we reject the null hypothesis and prove that there is correlation between bedroomcnt and logerror.
- After running a corralation test We reject the null hypothesis because there is a significant correlation between yearbuilt and logerror present.
- After running a corralation test we we fail to reject the null hypothesis because there is no significant correlation present between taxvaluedollarcnt and logerror.


4.Modeling:
- Conclusion for clusters 1, 2, & 3
- The OLS model outperformed (lassolars and polynomial regression) models with fractions of a decimal.
please reference project notebook for all model information)

Conclusion:
We were able to get close to beating the baseline but did not succeed at creating a better model

Suggestions:
looking further into the data for variables such as month, school zone, school district, retirement area, prior death in home,and using variables like Previous purchase price and date.




Summary of how we started :

--- Shape: (71858, 69)
--- Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 71858 entries, 0 to 71857
Data columns (total 69 columns):
 #   Column                        Non-Null Count  Dtype  
---  ------                        --------------  -----  
 0   propertylandusetypeid         71858 non-null  float64
 1   parcelid                      71858 non-null  int64  
 2   storytypeid                   47 non-null     float64
 3   typeconstructiontypeid        223 non-null    float64
 4   heatingorsystemtypeid         46680 non-null  float64
 5   buildingclasstypeid           0 non-null      float64
 6   architecturalstyletypeid      207 non-null    float64
 7   airconditioningtypeid         23069 non-null  float64
 8   id                            71858 non-null  int64  
 9   basementsqft                  47 non-null     float64
 10  bathroomcnt                   71858 non-null  float64
 11  bedroomcnt                    71858 non-null  float64
 12  buildingqualitytypeid         45063 non-null  float64
 13  calculatedbathnbr             71635 non-null  float64
 14  decktypeid                    589 non-null    float64
 15  finishedfloor1squarefeet      5697 non-null   float64
 16  calculatedfinishedsquarefeet  71702 non-null  float64
 17  finishedsquarefeet12          71523 non-null  float64
 18  finishedsquarefeet13          2 non-null      float64
 19  finishedsquarefeet15          11 non-null     float64
 20  finishedsquarefeet50          5697 non-null   float64
 21  finishedsquarefeet6           166 non-null    float64
 22  fips                          71858 non-null  float64
 23  fireplacecnt                  8104 non-null   float64
 24  fullbathcnt                   71635 non-null  float64
 25  garagecarcnt                  24971 non-null  float64
 26  garagetotalsqft               24971 non-null  float64
 27  hashottuborspa                1538 non-null   float64
 28  latitude                      71858 non-null  float64
 29  longitude                     71858 non-null  float64
 30  lotsizesquarefeet             63732 non-null  float64
 31  poolcnt                       15757 non-null  float64
 32  poolsizesum                   869 non-null    float64
 33  pooltypeid10                  464 non-null    float64
 34  pooltypeid2                   1074 non-null   float64
 35  pooltypeid7                   14665 non-null  float64
 36  propertycountylandusecode     71858 non-null  object 
 37  propertyzoningdesc            45480 non-null  object 
 38  rawcensustractandblock        71858 non-null  float64
 39  regionidcity                  70522 non-null  float64
 40  regionidcounty                71858 non-null  float64
 41  regionidneighborhood          28237 non-null  float64
 42  regionidzip                   71814 non-null  float64
 43  roomcnt                       71858 non-null  float64
 44  threequarterbathnbr           9999 non-null   float64
 45  unitcnt                       45436 non-null  float64
 46  yardbuildingsqft17            2250 non-null   float64
 47  yardbuildingsqft26            70 non-null     float64
 48  yearbuilt                     71667 non-null  float64
 49  numberofstories               17046 non-null  float64
 50  fireplaceflag                 172 non-null    float64
 51  structuretaxvaluedollarcnt    71764 non-null  float64
 52  taxvaluedollarcnt             71857 non-null  float64
 53  assessmentyear                71858 non-null  float64
 54  landtaxvaluedollarcnt         71856 non-null  float64
 55  taxamount                     71853 non-null  float64
 56  taxdelinquencyflag            2615 non-null   object 
 57  taxdelinquencyyear            2615 non-null   float64
 58  censustractandblock           71631 non-null  float64
 59  id.1                          71858 non-null  int64  
 60  logerror                      71858 non-null  float64
 61  transactiondate               71858 non-null  object 
 62  airconditioningdesc           23069 non-null  object 
 63  architecturalstyledesc        207 non-null    object 
 64  buildingclassdesc             0 non-null      float64
 65  heatingorsystemdesc           46680 non-null  object 
 66  typeconstructiondesc          223 non-null    object 
 67  storydesc                     47 non-null     object 
 68  propertylandusedesc           71858 non-null  object 
dtypes: float64(56), int64(3), object(10)
memory usage: 37.8+ MB
None
--- Descriptions:        propertylandusetypeid      parcelid  storytypeid  \
count           71858.000000  7.185800e+04         47.0   
mean              262.347908  1.304295e+07          7.0   
std                 2.217335  3.437438e+06          0.0   
min               261.000000  1.071186e+07          7.0   
25%               261.000000  1.153760e+07          7.0   
50%               261.000000  1.257360e+07          7.0   
75%               266.000000  1.425343e+07          7.0   
max               266.000000  1.676885e+08          7.0   

       typeconstructiontypeid  heatingorsystemtypeid  buildingclasstypeid  \
count              223.000000           46680.000000                  0.0   
mean                 6.040359               3.952421                  NaN   
std                  0.556035               3.657307                  NaN   
min                  4.000000               1.000000                  NaN   
25%                  6.000000               2.000000                  NaN   
50%                  6.000000               2.000000                  NaN   
75%                  6.000000               7.000000                  NaN   
max                 13.000000              24.000000                  NaN   

       architecturalstyletypeid  airconditioningtypeid            id  \
count                207.000000           23069.000000  7.185800e+04   
mean                   7.386473               1.867571  1.495823e+06   
std                    2.728030               3.062838  8.604787e+05   
min                    2.000000               1.000000  3.490000e+02   
25%                    7.000000               1.000000  7.545222e+05   
50%                    7.000000               1.000000  1.500115e+06   
75%                    7.000000               1.000000  2.240241e+06   
max                   21.000000              13.000000  2.982274e+06   

       basementsqft  ...  structuretaxvaluedollarcnt  taxvaluedollarcnt  \
count     47.000000  ...                7.176400e+04       7.185700e+04   
mean     678.978723  ...                1.886005e+05       4.908509e+05   
std      711.825226  ...                2.341052e+05       6.665600e+05   
min       38.000000  ...                4.400000e+01       1.000000e+03   
25%      263.500000  ...                8.311350e+04       2.038780e+05   
50%      512.000000  ...                1.348190e+05       3.570000e+05   
75%      809.500000  ...                2.165540e+05       5.677930e+05   
max     3560.000000  ...                1.142179e+07       4.906124e+07   

       assessmentyear  landtaxvaluedollarcnt      taxamount  \
count         71858.0           7.185600e+04   71853.000000   
mean           2016.0           3.024987e+05    5984.965270   
std               0.0           5.024066e+05    7777.210727   
min            2016.0           1.610000e+02      19.920000   
25%            2016.0           8.336850e+04    2679.620000   
50%            2016.0           2.036920e+05    4409.760000   
75%            2016.0           3.682615e+05    6866.620000   
max            2016.0           4.895220e+07  586639.300000   

       taxdelinquencyyear  censustractandblock          id.1      logerror  \
count         2615.000000         7.163100e+04  71858.000000  71858.000000   
mean            14.110899         6.050090e+13  38829.632678      0.016922   
std              2.223186         1.592079e+12  22385.329652      0.169059   
min              4.000000         6.037101e+13      0.000000     -4.655420   
25%             14.000000         6.037400e+13  19460.250000     -0.023629   
50%             15.000000         6.037621e+13  38849.500000      0.006647   
75%             15.000000         6.059052e+13  58211.750000      0.038436   
max             99.000000         4.830301e+14  77613.000000      5.262999   

       buildingclassdesc  
count                0.0  
mean                 NaN  
std                  NaN  
min                  NaN  
25%                  NaN  
50%                  NaN  
75%                  NaN  
max                  NaN  

[8 rows x 59 columns]
--- Nulls by Column: propertylandusetypeid               0
parcelid                            0
storytypeid                     71811
typeconstructiontypeid          71635
heatingorsystemtypeid           25178
buildingclasstypeid             71858
architecturalstyletypeid        71651
airconditioningtypeid           48789
id                                  0
basementsqft                    71811
bathroomcnt                         0
bedroomcnt                          0
buildingqualitytypeid           26795
calculatedbathnbr                 223
decktypeid                      71269
finishedfloor1squarefeet        66161
calculatedfinishedsquarefeet      156
finishedsquarefeet12              335
finishedsquarefeet13            71856
finishedsquarefeet15            71847
finishedsquarefeet50            66161
finishedsquarefeet6             71692
fips                                0
fireplacecnt                    63754
fullbathcnt                       223
garagecarcnt                    46887
garagetotalsqft                 46887
hashottuborspa                  70320
latitude                            0
longitude                           0
lotsizesquarefeet                8126
poolcnt                         56101
poolsizesum                     70989
pooltypeid10                    71394
pooltypeid2                     70784
pooltypeid7                     57193
propertycountylandusecode           0
propertyzoningdesc              26378
rawcensustractandblock              0
regionidcity                     1336
regionidcounty                      0
regionidneighborhood            43621
regionidzip                        44
roomcnt                             0
threequarterbathnbr             61859
unitcnt                         26422
yardbuildingsqft17              69608
yardbuildingsqft26              71788
yearbuilt                         191
numberofstories                 54812
fireplaceflag                   71686
structuretaxvaluedollarcnt         94
taxvaluedollarcnt                   1
assessmentyear                      0
landtaxvaluedollarcnt               2
taxamount                           5
taxdelinquencyflag              69243
taxdelinquencyyear              69243
censustractandblock               227
id.1                                0
logerror                            0
transactiondate                     0
airconditioningdesc             48789
architecturalstyledesc          71651
buildingclassdesc               71858
heatingorsystemdesc             25178
typeconstructiondesc            71635
storydesc                       71811
propertylandusedesc                 0
dtype: int64
nulls by row: n_missing  percent_missing
23         0.333333               2
24         0.347826              13
25         0.362319              24
26         0.376812              66
27         0.391304             312
28         0.405797             453
29         0.420290            5161
30         0.434783            3242
31         0.449275            9185
32         0.463768           11699
33         0.478261           14057
34         0.492754           12672
35         0.507246            4015
36         0.521739            5107
37         0.536232            3387
38         0.550725            1892
39         0.565217             240
40         0.579710             172
41         0.594203              13
42         0.608696               9
43         0.623188              13
44         0.637681              71
45         0.652174              45
46         0.666667               4
47         0.681159               2
48         0.695652               2
dtype: int64
None None None None None None


Preparing the data

1) Remove any properties that are likely to be something other than single unit properties. (e.g. no duplexes, no land/lot, ...). 
2) There are multiple ways to estimate that a property is a single unit, and there is not a single "right" answer.

Create a function that will drop rows or columns based on the percent of values that are missing:
handle_missing_values(df, prop_required_column, prop_required_row).

The input:
A dataframe
A number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column. 
i.e. if prop_required_column = .6, then you are requiring a column to have at least 60% of values not-NA (no more than 40% missing).
A number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row.
For example, if prop_required_row = .75, then you are requiring a row to have at least 75% of variables with a non-missing value (no more that 25% missing).
The output:
Look up the dropna documentation.
3) Encapsulate work is in function wrangle.py module.

columns that were removed and are in the original notebook

(id, id.1, parcelid,'propertylandusetypeid','buildingqualitytypeid'), id is not necessary for our algorithms and will confuse any models from here on forward.
(fullbathcnt,calculatedbathnbr,roomcnt), any room room count other the bedroomcnt or bathroomcnt is not necessary considering that they return similar information if not combined info.
(propertyzoningdesc,rawcensustractandblock,regionidcounty,censustractandblock), considering that fips is being kept for region identification purposes, these columns are not necessary.
(assessmentyear, landtaxvaluedollarcnt, taxamount, transactiondate), considering that we have already filtered out the data to only return back information for the year 2017, and are keeping taxvaluedollarcnt, these columns are not necessary because this information can be obtained through the data that we will be keeping.
(heatingorsystemdesc,finishedsquarefeet12,propertylandusedesc,'propertycountylandusecode','unitcnt'), calculatedfinishedsquarefeet already covers this info and heatingorsystemid already identifies this information numerically.


shortly after this we use wrangle.py and explore.py functions to run spliting and clustering commands so that we can visualize and attempt to make a good predcitive model.
