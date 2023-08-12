from collections import defaultdict

if __name__ == "__main__":
    with open(file='imageclasslist.txt', mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    lighting_types: defaultdict = defaultdict(int)

    print(len(lines[1:]))

    for line in lines[1:]:  # first line is definiton of .txt file not data needed
        lighting_types[int(line.split(' ')[2])] += 1

    print(sorted(lighting_types.items()))
    print(sum([v for k, v in lighting_types.items()]))
    print(lighting_types)
