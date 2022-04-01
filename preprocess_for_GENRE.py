import json


def main():
    with open('released_data/dev.traced.json', 'r') as f:
        data = json.load(f)