import pandas
import numpy

df_out_cont = pd.read_csv('allstateClaims_cont.csv', header=0, index_col=0)
df_out_cat = pd.read_csv('allstateClaims_cat.csv', header=0, index_col=0)
df_out_ids = pd.read_csv('allstateClaims_ids_out.csv', header=0, index_col=0)

# average the output of the two columns
out_cont = df_out_cont.values
out_cat = df_out_cat.values
out_ids = df_out_ids.values

output = (out_cont + out_cat) / 2.0

# write the output to csv
outputFile = 'allStateClaims_combination.csv'
print('writing output to %s...' % outputFile)
prediction_file = open(outputFile, 'w')
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(['id', 'loss'])
open_file_object.writerows(zip(ids_test, test_pred))
prediction_file.close()
print('completed')
