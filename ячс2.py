import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import pandas as pd
from sklearn.linear_model import LinearRegression


pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)



df = pd.read_excel(r'C:\Users\ilyx\Desktop\for py\Raschet_dokhoda_magazinchika (5) (3).xlsx')

years = [2017, 2018, 2019, 2020, 2021, 2022]






#ПРЕДСКАЗАНИЕ ПРИБЫЛИ ЗА 2023!!!!
k = 2017
zxc = []
zxc_ = []
model_predict = []
l = 0
for i in range(12):#12
    for j in range(6):

        zxc.append(df[k].iloc[l])
        x = np.array(years).reshape(-1, 1)
        y = np.array(zxc)
        k += 1
    model = LinearRegression()
    model.fit(x, y)
    model.coef_

    model_predict.append(model.predict([[2023]]))
    r_sq = model.score(x, y)
    df[2023].loc[l] = model.predict([[2023]])
    df['positive2023'].loc[l] = model.predict([[2023]]) + model.coef_
    df['2023negativ4ik'].loc[l] = model.predict([[2023]]) - model.coef_
    zxc = []
    l += 1
    k = 2017


years.append(2023)
#2024 year
k = 2017
zxc = []
l = 0
for i in range(12):#12
    for j in range(7):
        zxc.append(df[k].iloc[l])
        zxc_.append(df[k].iloc[l])
        k += 1

    x = np.array(years).reshape(-1, 1)
    y = np.array(zxc)
    model = LinearRegression()
    model.fit(x, y)
    model.coef_
    model_predict.append(model.predict([[2024]]))
    r_sq1 = model.score(x, y)
    df[2024].loc[l] = model.predict([[2024]])
    zxc_.append(df[2024].loc[l])
    df['positive2024'].loc[l] = model.predict([[2024]]) + model.coef_
    df['2024negativ4ik'].loc[l] = model.predict([[2024]]) - model.coef_
    # plt.scatter(years, zxc, alpha=0.4)

    zxc = []
    l += 1
    k = 2017



years.append(2024)


print(df*2)



fig, ax = plt.subplots()


# agg = [] # tochki vesni
# spring = []
# y1 = 2017
# d = 2
# for i in range(3):
#     for j in range(8):
#         spring.append(df[y1].loc[d])
#         y1 += 1
#     if i == 1:
#         x = np.array(years).reshape(-1, 1)
#         y = np.array(spring)
#         model = LinearRegression()
#         model.fit(x, y)
#         agg1 = (ax.plot(x, model.predict(x), color='red', lw=2, label = 'spring'))
#
#     agg.append(ax.scatter(np.array(years).reshape(-1, 1), spring, alpha=0.4))
#     spring = []
#     d += 1
#     y1 = 2017
agg = [] # tochki vesni
agg_summer = []
agg_autumn = []
agg_winter = []
winter = []
autumn = []
summer = []
spring = []
y1 = 2017
d = 0
for i in range(12):
    if 2 > i >=0:
        for j in range(8):
            winter.append(df[y1].loc[d])
            y1 += 1
        agg_winter.append(ax.scatter(np.array(years).reshape(-1, 1), winter, alpha=0.4))
        winter = []
        y1 = 2017
    if 5 > i >= 2:
        for j in range(8):
            spring.append(df[y1].loc[d])
            y1 += 1
        if i == 3:
            x = np.array(years).reshape(-1, 1)
            y = np.array(spring)
            model = LinearRegression()
            model.fit(x, y)
            agg1 = (ax.plot(x, model.predict(x), color='red', lw=2, label = 'spring'))

        agg.append(ax.scatter(np.array(years).reshape(-1, 1), spring, alpha=0.4))
        spring = []
        y1 = 2017
    if 8 > i >= 5:
        for j in range(8):
            summer.append(df[y1].loc[d])
            y1 += 1
        if i == 7:
            x = np.array(years).reshape(-1, 1)
            y = np.array(summer)
            model = LinearRegression()
            model.fit(x, y)
            agg_summer1 = (ax.plot(x, model.predict(x), color='red', lw=2, label = 'summer'))

        agg_summer.append(ax.scatter(np.array(years).reshape(-1, 1), summer, alpha=0.4))
        summer = []
        y1 = 2017
    if 11 > i >= 8:
        for j in range(8):
            autumn.append(df[y1].loc[d])
            y1 += 1
        if i == 9:
            x = np.array(years).reshape(-1, 1)
            y = np.array(autumn)
            model = LinearRegression()
            model.fit(x, y)
            agg_autumn1 = (ax.plot(x, model.predict(x), color='red', lw=2, label = 'autumn'))

        agg_autumn.append(ax.scatter(np.array(years).reshape(-1, 1), autumn, alpha=0.4))
        autumn = []
        y1 = 2017
    if i == 11:
        for j in range(8):
            winter.append(df[y1].loc[d])
            y1 += 1
        if i == 11:
            x = np.array(years).reshape(-1, 1)
            y = np.array(winter)
            model = LinearRegression()
            model.fit(x, y)
            agg_winter1 = (ax.plot(x, model.predict(x), color='red', lw=2, label = 'winter'))

        agg_winter.append(ax.scatter(np.array(years).reshape(-1, 1), winter, alpha=0.4))
        winter = []
        y1 = 2017
    d += 1
    y1 = 2017






















p2023 = []
neg2023 = []
p2024 = []
neg2024 = []
p = 0
for u in range(12):
    p1 = df['positive2023'].loc[p]
    p2 = df['2023negativ4ik'].loc[p]
    p3 = df['positive2024'].loc[p]
    p4 = df['2024negativ4ik'].loc[p]
    p2023.append(p1)
    neg2023.append(p2)
    p2024.append(p3)
    neg2024.append(p4)
    p += 1

k = 0
n2023 = []
n2024 = []
for j in range(12):
    n2023.append(df[2023].loc[k])
    n2024.append(df[2024].loc[k])
    k += 1



l0, = ax.plot(df.index, n2023, lw=2, label='neytralniy')
l1, = ax.plot(df.index, p2023, lw=2, label='pozitive', color='yellow')
l2, = ax.plot(df.index, neg2023, lw=2, label='negative')
p0, = ax.plot(df.index, n2024, lw=2, label='neytralniy')
p1, = ax.plot(df.index, p2024, lw=2, label='pozitive')
p2, = ax.plot(df.index, neg2024, lw=2, label='negative')

fig.subplots_adjust(left=0.25)
ax.grid(True)
plt.title('2023 and 2024')
fig.text(0.01, 0.77, '2023', fontsize = 15)
fig.text(0.01, 0.56, '2024', fontsize = 15)
fig.text(0.01, 0.33, 'seasons', fontsize = 15)
lines1 = [p0, p1, p2]
lines = [l0, l1, l2]



rax = fig.add_axes([0.01, 0.6, 0.145, 0.15])
labels = [str(line.get_label()) for line in lines]
visibility = [line.get_visible() for line in lines]
check = CheckButtons(rax, labels, visibility)

rax1 = fig.add_axes([0.01, 0.4, 0.145, 0.15])
labels1 = [str(line_.get_label()) for line_ in lines1]
visibility1 = [line_.get_visible() for line_ in lines1]
check1 = CheckButtons(rax1, labels1, visibility1)

rax2 = fig.add_axes([0.01, 0.25, 0.145, 0.05])
labels2 = [str(line_.get_label()) for line_ in agg and agg1]
visibility2 = [line_.get_visible() for line_ in agg and agg1]
check2 = CheckButtons(rax2, labels2, visibility2)

rax3 = fig.add_axes([0.01, 0.2, 0.145, 0.05])
labels3 = [str(line_.get_label()) for line_ in agg_summer and agg_summer1]
visibility3 = [line_.get_visible() for line_ in agg_summer and agg_summer1]
check3 = CheckButtons(rax3, labels3, visibility3)

rax4 = fig.add_axes([0.01, 0.15, 0.145, 0.05])
labels4 = [str(line_.get_label()) for line_ in agg_autumn and agg_autumn1]
visibility4 = [line_.get_visible() for line_ in agg_autumn and agg_autumn1]
check4 = CheckButtons(rax4, labels4, visibility4)

rax5 = fig.add_axes([0.01, 0.1, 0.145, 0.05])
labels5 = [str(line_.get_label()) for line_ in agg_winter and agg_winter1]
visibility5 = [line_.get_visible() for line_ in agg_winter and agg_winter1]
check5 = CheckButtons(rax5, labels5, visibility5)

ax.set_xlim([666, 777])

def func(label):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    ax.set_xlim([0, 11])
    ax.set_title('2023')
    plt.draw()
def func_(label):
    index1 = labels1.index(label)
    lines1[index1].set_visible(not lines1[index1].get_visible())
    ax.set_xlim([0, 11])
    ax.set_title('2024')
    plt.draw()
def spring_(label):
    index2 = labels2.index(label)
    agg1[index2].set_visible(not agg1[index2].get_visible())
    agg[index2].set_visible(not agg[index2].get_visible())
    index2 = labels2.index(label) + 1
    agg[index2].set_visible(not agg[index2].get_visible())
    index2 = labels2.index(label) + 2
    agg[index2].set_visible(not agg[index2].get_visible())
    ax.set_xlim([2016, 2025])
    ax.set_title('весна')
    plt.draw()

def summer_(label):
    index3 = labels3.index(label)
    agg_summer1[index3].set_visible(not agg_summer1[index3].get_visible())
    agg_summer[index3].set_visible(not agg_summer[index3].get_visible())
    index3 = labels3.index(label) + 1
    agg_summer[index3].set_visible(not agg_summer[index3].get_visible())
    index3 = labels3.index(label) + 2
    agg_summer[index3].set_visible(not agg_summer[index3].get_visible())
    ax.set_xlim([2016, 2025])
    ax.set_title('лето')
    plt.draw()

def autumn_(label):
    index4 = labels4.index(label)
    agg_autumn1[index4].set_visible(not agg_autumn1[index4].get_visible())
    agg_autumn[index4].set_visible(not agg_autumn[index4].get_visible())
    index4 = labels4.index(label) + 1
    agg_autumn[index4].set_visible(not agg_autumn[index4].get_visible())
    index4 = labels4.index(label) + 2
    agg_autumn[index4].set_visible(not agg_autumn[index4].get_visible())
    ax.set_xlim([2016, 2025])
    ax.set_title('осень')
    plt.draw()

def winter__(label):
    index5 = labels5.index(label)
    agg_winter1[index5].set_visible(not agg_winter1[index5].get_visible())
    agg_winter[index5].set_visible(not agg_winter[index5].get_visible())
    index5 = labels5.index(label) + 1
    agg_winter[index5].set_visible(not agg_winter[index5].get_visible())
    index5 = labels5.index(label) + 2
    agg_winter[index5].set_visible(not agg_winter[index5].get_visible())
    ax.set_xlim([2016, 2025])
    ax.set_title('зима')
    plt.draw()












check.on_clicked(func)
check1.on_clicked(func_)
check2.on_clicked(spring_)
check3.on_clicked(summer_)
check4.on_clicked(autumn_)
check5.on_clicked(winter__)
plt.show()




















