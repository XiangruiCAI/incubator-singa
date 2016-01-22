#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/

MSHADOW_FLAGS :=-DMSHADOW_USE_CUDA=0 -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0

libs :=singa glog protobuf
data_dname  = ckd_1year
emr_fname = emr_code
train_fname = emr_train
test_fname  = emr_test
numclass = 7 
dirshards = train_shard test_shard

create:
	protoc --proto_path=../../src/proto --proto_path=. --cpp_out=. mscnnlm.proto
	$(CXX) create_data.cc mscnnlm.pb.cc -std=c++11 -lsinga -lprotobuf -lzookeeper_mt -lglog -I../../include -I../../include/singa/proto \
		-L../../.libs/ -L/usr/local/lib -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/ \
		-o create_data.bin
	#for d in $(dirshards); do mkdir -p $${d}; done
	./create_data.bin -debug -emr $(data_dname)/$(emr_fname) -train $(data_dname)/$(train_fname) -test $(data_dname)/$(test_fname) -class_size $(numclass)

mscnnlm:
	protoc --proto_path=../../src/proto --proto_path=. --cpp_out=. mscnnlm.proto
	protoc --proto_path=../../src/proto --proto_path=. --python_out=../../tool/python/pb2 mscnnlm.proto
	$(CXX) main.cc mscnnlm.cc mscnnlm.pb.cc $(MSHADOW_FLAGS) -msse3 -std=c++11 -lsinga -lglog -lprotobuf -lopenblas -I../../include -I../../include/singa/proto \
		-L../../.libs/ -L/usr/local  -Wl,-unresolved-symbols=ignore-in-shared-libs -Wl,-rpath=../../.libs/\
		-o mscnnlm.bin
