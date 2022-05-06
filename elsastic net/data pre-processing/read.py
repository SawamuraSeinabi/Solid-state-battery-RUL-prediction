import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

train_data = pickle.load(open(r'.\Dataset\train_data.pkl', 'rb'))

# for i in range(len(train_data.keys())):
#     for j in range(1,101):
#         del train_data[i]['cycles'][j]['Qdlin']
#         del train_data[i]['cycles'][j]['Tdlin']
#         del train_data[i]['cycles'][j]['dQdV']
#         j = j + 1
#     i = i + 1
#
# savedir = './Dataset/train_data.pkl'
# with open(savedir, 'wb') as fp:
#     # 写入二进制
#     pickle.dump(train_data, fp)

for i in range(len(train_data.keys())):
#     for j in range(1, 100):
    dict_temp = train_data[i]['cycles'][10]
    df = pd.DataFrame(dict_temp)
    df = df[df['I'] < -0.05]
# df = df.drop_duplicates(subset=['I'])

    dict_temp2 = train_data[i]['cycles'][100]
    df2 = pd.DataFrame(dict_temp2)
    df2 = df2[df2['I'] < -0.05]
# df2 = df2.drop_duplicates(subset=['I'])


# plt.plot(df['t'], df['Qd'])
# plt.plot(df2['t'], df2['Qd'])
#
# # plt.axis([-0.01,0.005,2.0,3.5])
# plt.show()
# # plt.plot(df['t'], df['V'])
# # plt.plot(df['t'], df['I'])
# # plt.plot(df['t'], df['Qd'])
# # plt.legend(['V', 'I', 'Qd'])
# # plt.show()

    df_temp = df.drop_duplicates(subset=['V'])
    V_index = np.linspace(3.5, 2.0, 100)
    f = interp1d(df_temp['V'].values,df_temp['Qd'].values)
    temp1 = f(V_index)

    df2_temp = df2.drop_duplicates(subset=['V'])
    V_index = np.linspace(3.5, 2.0, 100)
    f = interp1d(df2_temp['V'].values,df2_temp['Qd'].values)
    temp2 = f(V_index)



    differences = temp2 - temp1
    plt.plot(differences, V_index)
plt.axis([-0.15, 0, 2.05, 3.5])
plt.xlabel('Q100-Q10(Ah)')
plt.ylabel('Voltage(V)')
plt.title('所有循环的Q100-Q10与电压曲线',fontsize=20)
plt.savefig(fname="delta Q2.svg", dpi=300, format="svg")
plt.show()