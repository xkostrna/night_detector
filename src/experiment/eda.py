"""
Exdark EDA

Manipulation with imageclasslist.txt

                    1          2 3 4 5
FORMAT of line: 2015_00001.png 1 2 1 1

1. filename
2. object class label
3. lightning type
4. indoor/outdoor
5. set
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

LIGHTNING_TYPES = {1: 'Low',
                   2: 'Ambient',
                   3: 'Object',
                   4: 'Single',
                   5: 'Weak',
                   6: 'Strong',
                   7: 'Screen',
                   8: 'Window',
                   9: 'Shadow',
                   10: 'Twilight'}

ENVIROMENT_TYPES = {1: 'Indoor',
                    2: 'Outdoor'}

SET_TYPES = {1: 'Training',
             2: 'Validation',
             3: 'Testing'}


def eda_analysis(imgclasslistpth: Path):
    """Iterate through all lines in imgclasslistpth and parse the lines into a dictionary"""
    lines = imgclasslistpth.read_text().splitlines()
    eda: {str, {str, defaultdict}} = {
        value: {key: defaultdict(int) for key in ['lightning_types', 'enviroment_types']}
        for value in SET_TYPES.values()
    }

    for line in lines:
        _, lghtng_type, env_type, set_type = [int(word) for word in line.split(' ') if word.isdigit()]
        eda[SET_TYPES[set_type]]['lightning_types'][LIGHTNING_TYPES[lghtng_type]] += 1
        eda[SET_TYPES[set_type]]['enviroment_types'][ENVIROMENT_TYPES[env_type]] += 1

    return eda


if __name__ == "__main__":
    # obtain eda analysis result
    result = eda_analysis(Path('imageclasslist.txt'))
    print(result)

    for set_type in SET_TYPES.values():
        lightning_types = {k: v for k, v in sorted(result[set_type]['lightning_types'].items())}

        # set plot width and height
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 10})  # Set the font size to 14

        # first subplot will be in grid of 1 row, 6 columns and will take space from 1st to 4th column
        plt.subplot(1, 6, (1, 4))  # noqa
        plt.bar(x=list(lightning_types.keys()),
                height=list(lightning_types.values()),
                color='g')

        enviroment_types = {k: v for k, v in sorted(result[set_type]['enviroment_types'].items())}
        plt.subplot(1, 6, (5, 6))  # noqa
        plt.bar(x=list(enviroment_types.keys()),
                height=list(enviroment_types.values()),
                color='b')

        plt.tight_layout()
        plt.show()
