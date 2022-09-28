import matplotlib.pyplot as plt
from collections import Counter
spsk = [5, 1, 3, 2, 3, 2, 2, 2, 3, 6, 6, 5, 4, 6, 2, 4, 6, 1, 1, 3]
print(*spsk)
# spsk = []
# for i in range(20):
#     inpt = int(input())
#     spsk.append(inpt)
spsk.sort()
chast = []
spsk1 = Counter(spsk).keys()
for number, count in Counter(spsk).items():
    if count > 1:
        chast.append(count)
spsk2 = list(spsk1)

print('Вариационный ряд')
print(*Counter(spsk))
# print(spsk1)
print(*chast)
# print(len(spsk))
j2 = []
vibor_sr = [l * r for (l, r) in zip(spsk1, chast)]
for r in spsk1:
    j1 = ((r - (sum(vibor_sr)/20))**2)
    j2.append(j1)
vibor_disp = [u * t for (u, t) in zip(j2, chast)]
print('Выборочное среднее: ', sum(vibor_sr)/20)
print('мода: ', max(Counter(spsk).values()))
print('Дисперсия ', sum(vibor_disp)/20)
print('Стандартное отклонение ', (sum(vibor_disp)/20)**0.5)
print('Кэф вариации', ((sum(vibor_disp)/20)**0.5)/(sum(vibor_sr)/20)*100)
print('размах: ', max(Counter(spsk)))
plt.axis([0,7,0,7]) #granitsi
plt.title('Полигон')
plt.xlabel('Варианты')
plt.ylabel('Частота')
plt.plot(spsk1,chast,'ro')
plt.show()
plt.axis([0,7,0,7]) #granitsi
plt.title('Гистрограмма')
plt.xlabel('Варианты')
plt.ylabel('Частота')
plt.bar(spsk1,chast)
plt.show()