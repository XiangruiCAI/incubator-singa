import random

ratio = 0.9
train = open("emr_train", "w")
test = open("emr_test", "w")
f = open("emr_code.bak", "r")
emr = open("emr_code", "w")

stat = ('N18 ', 'N181', 'N182', 'N183', 'N184', 'N185', 'N189')
stat_num = [0,0,0,0,0,0,0]

c = []
for i in range(len(stat)):
    c.append(open(stat[i], "w"))

data = f.read().split('\n')

part = [[],[],[],[],[],[],[]]
for item in data:
    for i in range(len(stat)):
        if stat[i] in item:
            c[i].write(item + '\n')
            part[i].append(item)
            stat_num[i] += 1
            break

for i in range(len(stat)):
    print stat[i] + ": " + str(stat_num[i])

res = []

for i in range(1, len(stat) - 1):
    res.extend(part[i])

for item in res:
    item = item.replace('N181', 'N0').replace('N182', 'N0').replace('N183', 'N0').replace('N184', 'N1').replace('N185', 'N1')
    emr.write(item + '\n')

split = int(len(res) * ratio)
random.shuffle(res)

train_data = res[:split]
test_data = res[split:]

for item in train_data:
    item = item.replace('N181', 'N0').replace('N182', 'N0').replace('N183', 'N0').replace('N184', 'N1').replace('N185', 'N1')
    train.write(item + '\n')
for item in test_data:
    item = item.replace('N181', 'N0').replace('N182', 'N0').replace('N183', 'N0').replace('N184', 'N1').replace('N185', 'N1')
    test.write(item + '\n')

f.close()
emr.close()
train.close()
test.close()
