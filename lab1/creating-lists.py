import random

numbers_300 = []
with open('./lab1/числа для варианта (Артем).txt', 'r') as f:
# with open('./lab1/числа для варианта (Рома).txt', 'r') as f:
# with open('./lab1/числа для варианта (Саня).txt', 'r') as f:
    for line in f:
        numbers_300.append(float(line.strip()))
random_10 = random.sample(numbers_300, 10)
random_20 = random.sample(numbers_300, 20)
random_50 = random.sample(numbers_300, 50)
random_100 = random.sample(numbers_300, 100)
random_200 = random.sample(numbers_300, 200)
with open('./lab1/10.txt', 'w') as f:
    for number in random_10:
        f.write(str(number) + '\n')
with open('./lab1/20.txt', 'w') as f:
    for number in random_20:
        f.write(str(number) + '\n')
with open('./lab1/50.txt', 'w') as f:
    for number in random_50:
        f.write(str(number) + '\n')
with open('./lab1/100.txt', 'w') as f:
    for number in random_100:
        f.write(str(number) + '\n')
with open('./lab1/200.txt', 'w') as f:
    for number in random_200:
        f.write(str(number) + '\n')
with open('./lab1/300.txt', 'w') as f:
    for number in numbers_300:
        f.write(str(number) + '\n')