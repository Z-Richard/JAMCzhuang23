# JAMCzhuang24

## Data Files

`event.csv` specify the precipitation events used in the training data.

| Column Name  | Description |
| -----------  | ----------- |
| station      | the 3-digit station code for each METAR station |
| start_time   | the start time of the precipitation episode |
| end_time     | the end time of the precipitation episode |
| lon          | station longitude |
| lat          | station latitude |
| median_time  | the median time of the precipitation episode |
| year         | the year in which the precipitation episode occurs |
| month        | the month in which the precipitation episode occurs |
| label        | '0' - rain, '1' - snow, '2' - freezing rain, '3' - ice pellets |

`US_X_30_30_2.5_1.3_0.25_constrained.npy` and `US_y_30_30_2.5_1.3_0.25_constrained.npy` can be used to train the LightGBMClassifier. In particular, 

| Column Number  | Description |
| -------------  | ----------- |
| 0              | Surface Temperature (K) |
| 1 - 16         | 1000 - 500 hPa Temperature (K) |
| 17             | Surface Relative Humidity (%) |
| 18 - 33        | 1000 - 500 hPa Relative Humidity (%) |

We note that `30_30_2.5_1.3` means that 30000 rain events, 30000 snow events, 2500 freezing rain events, and 1300 ice pellets events are used to train the algorithm. `0.25` denotes a horizontal resolution of 0.25 degree. `constrained` means that relative humidity is constrained to between 0 and 100%. 

`US_X_30_30_2.5_1.3_1deg_constrained.npy` and `US_y_30_30_2.5_1.3_1deg_constrained.npy` are analogous to their 0.25 deg counterparts but should be used on model data with a horizontal resolution of about 1 degree. 

## Example

```
import numpy as np
import lightgbm as lgb

X = np.load('US_X_30_30_2.5_1.3_0.25_constrained.npy')
y = np.load('US_y_30_30_2.5_1.3_0.25_constrained.npy')

clf = lgb.LGBMClassifier(num_leaves=27,
                         learning_rate=0.05,
                         max_depth=6,
                         min_child_samples=18, 
                         max_bins=300,
                         verbosity=-1,
                         data_sample_strategy='goss',
                         importance_type='gain')
                             
clf.predict(X_test)  # X_test should have a shape of (N, 34)
```
