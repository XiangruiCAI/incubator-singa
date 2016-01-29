#! /usr/bin/python
import random

f = open("emr_code", "r")
o = open("emr_new", "w")

a = 0
b = 0
line = []

for item in f.readlines():
    if "N18 " in item:
        line.append(item.replace("N18", "N0"))
    if "N181" in item:
        line.append(item.replace("N181", "N0"))
    if "N182" in item:
        line.append(item.replace("N182", "N0"))
    if "N183" in item:
        line.append(item.replace("N183", "N0"))
    if "N184" in item:
        line.append(item.replace("N184", "N1"))
    if "N185" in item:
        line.append(item.replace("N185", "N1"))

ratio = 0.9
train = open("emr_train", "w")
test = open("emr_test", "w")

split = int(len(line) * ratio)
random.shuffle(line)

for item in line:
    o.write(item)
    if "N0" in item:
        a += 1
    else:
        b += 1

print "N0: " + str(a)
print "N1: " + str(b)

train_data = line[:split]
test_data = line[split:]

for item in train_data:
    train.write(item)
for item in test_data:
    test.write(item)

f.close()
o.close()
train.close()
test.close()
