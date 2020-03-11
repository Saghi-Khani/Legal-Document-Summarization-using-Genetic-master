import csv
import collections

file= 'results.csv'
def unique():
    rows = list(csv.reader(open(file, 'r'), delimiter=','))
    result = collections.OrderedDict()
    for r in rows:
        key = (r[0])  ## The pair (r[1],r[6]) must be unique
        if key not in result:
            result[key] = r

    return result.values()

result = unique()

file = 'cleaned_result.csv'

with open(file,'w',newline='') as out:
	writer = csv.writer(out)
	for item in list(result):
		# print()
		writer.writerow(item) 
	out.close()
