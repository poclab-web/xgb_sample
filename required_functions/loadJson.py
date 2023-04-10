import json

def loadJson(json_path):
    f = open(json_path, 'r')
    return json.load(f)
