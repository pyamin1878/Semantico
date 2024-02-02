import json


file_path = 'wiki_corpus.json'

try:
    with open(file_path, 'r') as file:
        data = json.load(file)
        print(data) 
except FileNotFoundError:
    print("File not found. Please check the file path.")
except json.JSONDecodeError:
    print("Error parsing JSON. Please check the file's format.")