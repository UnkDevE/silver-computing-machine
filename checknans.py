#!/bin/python
import json

with open("featureset_test_kmeans.json", "r") as r:
    red = json.load(r)
    if [r[1] for r in red if str(r[1]) != 'nan'] == []:
        print("SUCCESS")
    else:
        print("FAIL")
