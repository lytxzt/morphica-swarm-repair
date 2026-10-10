import random
import pandas as pd
import numpy as np
import Utils

N = 2000 # 假设需要生成10个无人机的坐标
D = 3200 # 假设D为50

connected = False

while not connected:
    data = []
    for _ in range(N):
        x = random.randint(1, D)
        y = random.randint(1, D)
        z = random.randint(1, 100)
        data.append([x, y, z])

    data_np = np.array(data)
    connected, _ = Utils.check_if_a_connected_graph(data_np, len(data_np))
    print(connected)

df = pd.DataFrame(data, columns=['x', 'y', 'z'])
df.to_excel('drone_positions.xlsx', index=False)