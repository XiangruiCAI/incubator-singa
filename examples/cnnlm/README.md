An extreamly simple model for disease classification.

### Introduction
  Originally, we wanted to classify the patients into 7 classes(N18, N181, N182, N183, N184, N185, N189).
  But the result was very bias and weird. All the results belonged to one class.
  So I tried two attempts below, hoping to see relative good results. Unfortunately, I failed again.

### Model architecture
  Besides, to ensure whether the reason is my coding error, I remove user-defined layer as many as possible.
  So I remove embedding layer, ignore delta time and use built-in cnn and pooling.
  Now, the setting is as follows:
  * data layer, with one-hot vectors.
  * first convolution layer
  * first pooling layer
  * second convolution layer
  * second pooling layer
  * inner product layer
  * softmax loss layer

### Input data
  1. Use balanced samples.
  I choose N183, N184, N185, N189, each class has 650 samples.
  Result is still bad.

  2. Classify the patients with kidney disease into 2 classes: mild and severe. 
  N181 + N182 + N183 --> mild (1186 samples)
  N184 + N185 --> severe (2926 samples)
  Result is still bad.

  In this folder, I include the second data.

### Data statistics
  Please run python script to see the statistics
  python ./ckd_1year/stat.py

### Compile
  make create // for data creation
  maek cnnlm  // for this simple model
