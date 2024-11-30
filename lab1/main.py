import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

# График 1
# Распределение случайных величин (y - числа, x - номер числа)

plt.gca().set_facecolor("lightgray")
plt.plot(numbers_300, color="red")
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.grid(True, linewidth=0.5, color="white")
plt.savefig("lab1/1.png")

# График 2
# Гистограмма распределения случайных величин
plt.clf()
plt.gca().set_facecolor("lightgray")
plt.hist(numbers_300, bins=15, color="red", edgecolor="black", linewidth=1, zorder=2)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.grid(True, linewidth=0.5, color="white", zorder=0)
plt.savefig("lab1/2.png")

# График 3 - Гистограмма для 300 случайных чисел с распределением Эрланга и 300 чисел из заданного ЧП
shape_param = 2  # Параметр k для Эрланга
scale_param = d_best / m_best  # Параметр λ для Эрланга
erlang_300 = np.random.gamma(shape=shape_param, scale=scale_param, size=300).tolist()
erlang_300_to_file = open("./lab1/aprox.txt","w")
for x in erlang_300:
    erlang_300_to_file.write(str(x)+'\n')
erlang_300_to_file.close()
plt.clf()
plt.gca().set_facecolor("lightgray")
plt.hist(
    numbers_300,
    color="blue",
    label="numbers_300",
    bins=15,
    linewidth=1,
    zorder=2,
    alpha=0.5,
)
plt.hist(
    erlang_300,
    color="red",
    alpha=0.5,
    label="erlang_300",
    bins=15,
    linewidth=1,
    zorder=2,
)
plt.grid(True, linewidth=0.5, color="white", zorder=0)
plt.xlabel("Значение")
plt.ylabel("Частота")
plt.legend()
plt.savefig("lab1/3.png")

# Генерируем последовательность случайных чисел с распределением Эрланга
erlang_200 = np.random.gamma(shape=shape_param, scale=scale_param, size=200).tolist()
erlang_100 = np.random.gamma(shape=shape_param, scale=scale_param, size=100).tolist()
erlang_50 = np.random.gamma(shape=shape_param, scale=scale_param, size=50).tolist()
erlang_20 = np.random.gamma(shape=shape_param, scale=scale_param, size=20).tolist()
erlang_10 = np.random.gamma(shape=shape_param, scale=scale_param, size=10).tolist()
m_300 = sum(erlang_300) / len(erlang_300)
d_300 = (sum([x**2 for x in erlang_300]) / 300 - m_300**2) * (300 / 299)
msd_300 = d_300**0.5
sigma_m_300 = (d_300 / 300) ** 0.5
e_p1_300 = t_p[0] * sigma_m_300
e_p2_300 = t_p[1] * sigma_m_300
e_p3_300 = t_p[2] * sigma_m_300
variation_300 = msd_300 / m_300
print()
print("-" * 20)
print("erlang 300")
print("m =", round(m_300, 3))
print("d =", round(d_300, 3))
print("msd =", round(msd_300, 3))
print("sigma_m =", round(sigma_m_300, 3))
print("e_p1 =", round(e_p1_300, 3))
print("e_p2 =", round(e_p2_300, 3))
print("e_p3 =", round(e_p3_300, 3))
print("variation =", round(variation_300, 3))

new_smalls = [erlang_10, erlang_20, erlang_50, erlang_100, erlang_200]
m_new = []
d_new = []
msd_new = []
sigma_m_new = []
e_p1_new = []
e_p2_new = []
e_p3_new = []
variation_new = []
m_div_new = []
d_div_new = []
msd_div_new = []
e_p1_div_new = []
e_p2_div_new = []
e_p3_div_new = []
variation_div_new = []

for i in range(5):
    m_new.append(sum(new_smalls[i]) / len(new_smalls[i]))
    d_new.append(
        (sum([x**2 for x in new_smalls[i]]) / len(new_smalls[i]) - m_new[i] ** 2)
        * (len(new_smalls[i]) / (len(new_smalls[i]) - 1))
    )
    msd_new.append(d_new[i] ** 0.5)
    sigma_m_new.append((d_new[i] / len(new_smalls[i])) ** 0.5)
    e_p1_new.append(t_p[0] * sigma_m_new[i])
    e_p2_new.append(t_p[1] * sigma_m_new[i])
    e_p3_new.append(t_p[2] * sigma_m_new[i])
    m_div_new.append(abs((m[i] - m_new[i]) / m[i] * 100))
    d_div_new.append(abs((d[i] - d_new[i]) / d[i] * 100))
    msd_div_new.append(abs((msd[i] - msd_new[i]) / msd[i] * 100))
    e_p1_div_new.append(abs((e_p1[i] - e_p1_new[i]) / e_p1[i] * 100))
    e_p2_div_new.append(abs((e_p2[i] - e_p2_new[i]) / e_p2[i] * 100))
    e_p3_div_new.append(abs((e_p3[i] - e_p3_new[i]) / e_p3[i] * 100))
    variation_new.append(msd_new[i] / m_new[i])
    variation_div_new.append(
        abs((variation[i] - variation_new[i]) / variation[i] * 100)
    )

m_new = list(map(round3, m_new))
d_new = list(map(round3, d_new))
msd_new = list(map(round3, msd_new))
sigma_m_new = list(map(round3, sigma_m_new))
e_p1_new = list(map(round3, e_p1_new))
e_p2_new = list(map(round3, e_p2_new))
e_p3_new = list(map(round3, e_p3_new))
m_div_new = list(map(round3, m_div_new))
d_div_new = list(map(round3, d_div_new))
msd_div_new = list(map(round3, msd_div_new))
e_p1_div_new = list(map(round3, e_p1_div_new))
e_p2_div_new = list(map(round3, e_p2_div_new))
e_p3_div_new = list(map(round3, e_p3_div_new))
variation_new = list(map(round3, variation_new))
variation_div_new = list(map(round3, variation_div_new))
print("M: ", m_new)
print("D: ", d_new)
print("MSD: ", msd_new)
print("E_p1: ", e_p1_new)
print("E_p2: ", e_p2_new)
print("E_p3: ", e_p3_new)
print("M div: ", m_div_new)
print("D div: ", d_div_new)
print("MSD div: ", msd_div_new)
print("E_p1 div: ", e_p1_div_new)
print("E_p2 div: ", e_p2_div_new)
print("E_p3 div: ", e_p3_div_new)
print("Variation: ", variation_new)
print("Variation div: ", variation_div_new)


# Рассчёт коэффициентов автокорреляции для numbers_300. Вывод графика
lags = range(1, 11)
n = len(numbers_300)
numbers_autocorrs = []
for lag in range(1, 11):
    numerator = np.sum(
        (numbers_300[:-lag] - np.mean(numbers_300[:-lag]))
        * (numbers_300[lag:] - np.mean(numbers_300[lag:]))
    )
    denominator = np.sum((numbers_300[:-lag] - np.mean(numbers_300[:-lag])) ** 2)
    autocorr = numerator / denominator
    numbers_autocorrs.append(autocorr)
print("\nAutocorrs number_300: ", list(float(i) for i in numbers_autocorrs))
plt.clf()
plt.gca().set_facecolor("lightgray")
plt.plot(lags, numbers_autocorrs, "r-", marker="o")
plt.xlabel("Сдвиг")
plt.ylabel("Коэффициент автокорреляции")
plt.grid(True, linewidth=0.5, color="white")
plt.savefig("lab1/4.png")


# Рассчёт коэффициентов автокорреляции для erlang_300. Вывод графика
erlang_autocorrs = []
for lag in range(1, 11):
    numerator = np.sum(
        (erlang_300[:-lag] - np.mean(erlang_300[:-lag]))
        * (erlang_300[lag:] - np.mean(erlang_300[lag:]))
    )
    denominator = np.sum((erlang_300[:-lag] - np.mean(erlang_300[:-lag])) ** 2)
    autocorr = numerator / denominator
    erlang_autocorrs.append(autocorr)
print("Autocorrs erlang_300: ", list(float(i) for i in erlang_autocorrs))
plt.clf()
plt.gca().set_facecolor("lightgray")
plt.plot(lags, erlang_autocorrs, "r-", marker="o")
plt.xlabel("Сдвиг")
plt.ylabel("Коэффициент автокорреляции")
plt.grid(True, linewidth=0.5, color="white")
plt.savefig("lab1/5.png")

# сравнение плотности распределения аппроксимирующего закона с гистограммой распределения частот для исходной ЧП
plt.clf()
plt.gca().set_facecolor("lightgray")
plt.hist(
    numbers_300,
    bins=15,
    density=True,
    color="red",
    edgecolor="black",
    linewidth=1,
    zorder=2,
    label="Исходные данные",
)
plt.xlabel("Значение")
plt.ylabel("Плотность распределения")
plt.grid(True, linewidth=0.5, color="white", zorder=0)
exp_lambda = 1 / np.mean(numbers_300)
x = np.linspace(0, max(numbers_300), 1000)
exp_pdf = exp_lambda * np.exp(-exp_lambda * x)
erlang_lambda = 2 / np.mean(numbers_300)
erlang_pdf = stats.erlang.pdf(x, a=2, scale=1 / erlang_lambda)
plt.plot(x, erlang_pdf, "b-", lw=2, label="Распределение Эрланга (k=2)")
plt.legend()
plt.savefig("lab1/6.png")
