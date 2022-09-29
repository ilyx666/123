import matplotlib.pyplot as plt
from collections import Counter
# spsk = []
# while len(spsk) < 20:
#     try:
#         inp = int(input('>>>'))
#         spsk.append(inp)
#     except ValueError:
#         print('Только числа(целые)')
spsk = [5, 1, 3, 2, 3, 2, 2, 2, 3, 6, 6, 5, 4, 6, 2, 4, 6, 1, 1, 3]
print(*spsk)

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
print('Кэф вариации', ((sum(vibor_disp)/20)**0.5)/(sum(vibor_sr)/20))
print('размах: ', max(spsk)-min(spsk))
print('Медиана ', (spsk[9]+spsk[10])//2)


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





#2 ZADANIE
print('--2 zadanie--')
spsk = [5, 1, 3, 2, 3, 2, 2, 2, 3, 6, 6, 5, 4, 6, 2, 4, 6, 1, 1, 3]
print('прибыль за последние 20 недель: ', *spsk)
print('заплонированная прибыль: 3,5 тыс. в неделю')
spsk.sort()
chast = []
spsk1 = Counter(spsk).keys()
for number, count in Counter(spsk).items():
    if count > 1:
        chast.append(count)
spsk2 = list(spsk1)
j2 = []
vibor_sr = [l * r for (l, r) in zip(spsk1, chast)]
for r in spsk1:
    j1 = ((r - (sum(vibor_sr)/20))**2)
    j2.append(j1)
vibor_disp = [u * t for (u, t) in zip(j2, chast)]
# print('Выборочное среднее: ', sum(vibor_sr)/20)
# print('Стандартное отклонение ', (sum(vibor_disp)/20)**0.5)
g1 = sum(vibor_sr)/20 - (3*(sum(vibor_disp)/20)**0.5)
g2 = sum(vibor_sr)/20 + (3*(sum(vibor_disp)/20)**0.5)
g3 = float(7/2)
if g1 <= g3 <= g2:
    print(g1, '< 3,5 <', g2,'условие соблюдается')
else:
    print(g1, '< 3,5 <', g2, 'условие не соблюдается')

