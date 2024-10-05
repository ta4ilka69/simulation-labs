numbers_300, numbers_10, numbers_20, numbers_50, numbers_100, numbers_200 = (
    [],
    [],
    [],
    [],
    [],
    [],
)
with open("./lab1/300.txt", "r") as f:
    for line in f:
        numbers_300.append(float(line))
with open("./lab1/10.txt", "r") as f:
    for line in f:
        numbers_10.append(float(line))
with open("./lab1/20.txt", "r") as f:
    for line in f:
        numbers_20.append(float(line))
with open("./lab1/50.txt", "r") as f:
    for line in f:
        numbers_50.append(float(line))
with open("./lab1/100.txt", "r") as f:
    for line in f:
        numbers_100.append(float(line))
with open("./lab1/200.txt", "r") as f:
    for line in f:
        numbers_200.append(float(line))

not_best = [numbers_10, numbers_20, numbers_50, numbers_100, numbers_200]

t_p = [1.643, 1.960, 2.576]

# best case

# m - мат. ожидание
# d - дисперсия
# msd - среднеквадратическое отклонение
# sigma_m - среднеквадратическое отклонение среднего, оно же сигма m
# e_p1, e_p2, e_p3 - эпсилон_p из презентации для доверительного интервала (m = m_оценка +- e_p)
# где e_p = t_p * sigma_m
# variation - коэффициент вариации

m_best = sum(numbers_300) / len(numbers_300)
d_best = (sum([x**2 for x in numbers_300]) / 300 - m_best**2) * (300 / 299)
msd_best = d_best**0.5
sigma_m_best = (d_best / 300) ** 0.5
e_p1_best = t_p[0] * sigma_m_best
e_p2_best = t_p[1] * sigma_m_best
e_p3_best = t_p[2] * sigma_m_best
variation_best = msd_best / m_best
print("Best case")
print("m =", round(m_best, 3))
print("d =", round(d_best, 3))
print("msd =", round(msd_best, 3))
print("sigma_m =", round(sigma_m_best, 3))
print("e_p1 =", round(e_p1_best, 3))
print("e_p2 =", round(e_p2_best, 3))
print("e_p3 =", round(e_p3_best, 3))
print("variation =", round(variation_best, 3))

# other cases

m = []
d = []
m_div = []
d_div = []
msd = []
sigma_m = []
e_p1 = []
e_p2 = []
e_p3 = []
msd_div = []
e_p1_div = []
e_p2_div = []
e_p3_div = []
variation = []
variation_div = []
for i in range(5):
    m.append(sum(not_best[i]) / len(not_best[i]))
    d.append(
        (sum([x**2 for x in not_best[i]]) / len(not_best[i]) - m[i] ** 2)
        * (len(not_best[i]) / (len(not_best[i]) - 1))
    )
    msd.append(d[i] ** 0.5)
    sigma_m.append((d[i] / len(not_best[i])) ** 0.5)
    e_p1.append(t_p[0] * sigma_m[i])
    e_p2.append(t_p[1] * sigma_m[i])
    e_p3.append(t_p[2] * sigma_m[i])
    m_div.append(abs((m_best - m[i]) / m_best * 100))
    d_div.append(abs((d_best - d[i]) / d_best * 100))
    msd_div.append(abs((msd_best - msd[i]) / msd_best * 100))
    e_p1_div.append(abs((e_p1_best - e_p1[i]) / e_p1_best * 100))
    e_p2_div.append(abs((e_p2_best - e_p2[i]) / e_p2_best * 100))
    e_p3_div.append(abs((e_p3_best - e_p3[i]) / e_p3_best * 100))
    variation.append(msd[i] / m[i])
    variation_div.append(abs((variation_best - variation[i]) / variation_best * 100))

def round3(n):
    return round(n, 3)

m = list(map(round3, m))
d = list(map(round3, d))
msd = list(map(round3, msd))
sigma_m = list(map(round3, sigma_m))
e_p1 = list(map(round3, e_p1))
e_p2 = list(map(round3, e_p2))
e_p3 = list(map(round3, e_p3))
m_div = list(map(round3, m_div))
d_div = list(map(round3, d_div))
msd_div = list(map(round3, msd_div))
e_p1_div = list(map(round3, e_p1_div))
e_p2_div = list(map(round3, e_p2_div))
e_p3_div = list(map(round3, e_p3_div))
variation = list(map(round3, variation))
variation_div = list(map(round3, variation_div))
for i, N in enumerate([10, 20, 50, 100, 200]):
    print("-" * 20)
    print()
    print(f"N = {N}")
    print("m =", m[i])
    print("d =", d[i])
    print("msd =", msd[i])
    print("sigma_m =", sigma_m[i])
    print("e_p1 =", e_p1[i])
    print("e_p2 =", e_p2[i])
    print("e_p3 =", e_p3[i])
    print("difference m =", m_div[i])
    print("difference d =", d_div[i])
    print("difference msd =", msd_div[i])
    print("difference e_p1 =", e_p1_div[i])
    print("difference e_p2 =", e_p2_div[i])
    print("difference e_p3 =", e_p3_div[i])
    print("variation =", variation[i])
    print("difference variation =", variation_div[i])


#TODO проверить формулу для к-та вариации с https://ru.wikipedia.org/wiki/Коэффициент_вариации
#TODO https://studfile.net/preview/9173159/page:24/ Если НЕ верно, то прикинуть, какой закон распределения подходит
#TODO сейчас - Логарифмически-нормальное 0.35-0.80 ;  у нас 0.4-0.7