import pandas as pd
import numpy as np

with open('items.txt', 'w') as f:
    cbb = pd.read_csv('src\dataTableInputs\websiteCBB.csv')
    print(cbb.iloc[0])
    print(cbb.iloc[0][0])
    for i in range(len(cbb)):
        row = cbb.iloc[i]
        f.write(f"{{team: '{row[0]}', EFGO: '{row[1]}', EFGD: '{row[2]}', TOR: '{row[3]}', TORD: '{row[4]}', ORB: '{row[5]}', DRB: '{row[6]}', FTR: '{row[7]}', FTRD: '{row[8]}', YEAR: '{row[9]}', WR: '{row[10]}'}},")
        f.write("\n")
    f.close()