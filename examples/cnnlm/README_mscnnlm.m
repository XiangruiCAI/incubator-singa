exacmples/mscnnlm

1.
read emr_code, and
generate train.bin and test.bin
note that emr_code is divided into emr_train and emr_test

$ make create

2.
compile mscnnlm.cc

$ make mscnnlm

3.
revise job.conf
currently we have only data layer


4. runsinga

at Singa_Root

$ bin/singa-run.sh -exec examples/mscnnlm/mscnnlm.bin -conf examples/mscnnlm/job.conf


