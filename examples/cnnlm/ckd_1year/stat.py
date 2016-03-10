#!/usr/bin/python
#encoding=utf-8

def average(l):
    return sum(l) / float(len(l))

def variance(l):
    square_sum = 0.0
    for item in l:
        square_sum += item * item

    n = float(len(l))
    avg = average(l)
    return square_sum / n - avg * avg

def LE(l, th):
    cnt = 0
    for item in l:
        if item <= th:
            cnt += 1
    return cnt

l_num_words = []
l_num_chars = []
with open("emr_code.bak", "rb") as f:
    for line in f:
        words = line.split(' ')
        l_num_words.append(len(words) / 2)
        for i in range(1, len(words)):
            if i % 2 == 0:
                continue
            l_num_chars.append(words[i].count('c'))

print "============word statistics=========="
print "average of num of words: %f" %average(l_num_words)
print "variance of num of words: %f" %variance(l_num_words)
print "max of num of words: %d" %max(l_num_words)
print "min of num of words: %d" %min(l_num_words)
for i in range (1, max(l_num_words) / 100 + 2):
    print "num of words less than %d: %d" %(i * 100, LE(l_num_words, i * 100))
print "============char statistics=========="
print "average of num of chars: %f" %average(l_num_chars)
print "variance of num of chars: %f" %variance(l_num_chars)
print "max of num of chars: %d" %max(l_num_chars)
print "min of num of chars: %d" %min(l_num_chars)
for i in range (1, max(l_num_chars) / 10 + 2):
    print "num of chars less than %d: %d" %(i * 10, LE(l_num_chars, i * 10))

