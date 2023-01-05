
import pandas as pd


cc_apps = pd.read_csv('Datasets/cc_approvals.data', header= None)
print(cc_apps.head())


cc_apps_description = cc_apps.describe()
print(cc_apps_description)
print('\n')

cc_apps_info = cc_apps.info()
print(cc_apps_info)
print('\n')

print(cc_apps.tail(17))

