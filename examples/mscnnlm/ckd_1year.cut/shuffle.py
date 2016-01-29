import random

ratio = 0.9
train = open("emr_train", "w")
test = open("emr_test", "w")

stat = ('N18 ', 'N181', 'N182', 'N183', 'N184', 'N185', 'N189')
stat_num = [0,0,0,0,0,0,0]
with open("emr_code", "rb") as f:
    data = f.read().split('\n')

for item in data:
    for i in range(len(stat)):
        if stat[i] in item:
            stat_num[i] += 1
            break

for i in range(len(stat)):
    print stat[i] + ': ' + str(stat_num[i])

split = int(len(data) * ratio)
random.shuffle(data)

train_data = data[:split]
test_data = data[split:]

for item in train_data:
    train.write(item + '\n')
for item in test_data:
    test.write(item + '\n')

f.close()
train.close()
test.close()
