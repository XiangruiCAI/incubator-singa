/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/
#include "./mscnnlm.h"

#include <string>
#include <algorithm>
#include <glog/logging.h>
#include "mshadow/tensor.h"
#include "mshadow/tensor_expr.h"
#include "mshadow/cxxnet_op.h"
#include "./mscnnlm.pb.h"

namespace mscnnlm {
using std::vector;
using std::string;

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Tensor;

inline Tensor<cpu, 2> RTensor2(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> RTensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}

/*******DataLayer**************/
DataLayer::~DataLayer() {
  if (store_ != nullptr)
    delete store_;
}

void DataLayer::Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Setup @ Data";
  MSCNNLayer::Setup(conf, srclayers);
  string key;
  //max_window_ = conf.GetExtension(data_conf).max_window();
  max_word_len_ = conf.GetExtension(data_conf).max_word_len(); // max num chars in a word
  max_num_word_ = conf.GetExtension(data_conf).max_num_word();
  batchsize_ = conf.GetExtension(data_conf).batchsize();
  vacab_size_ = conf.GetExtension(data_conf).vacab_size();
  data_.Reshape(vector<int>{batchsize_, max_word_len_ * max_num_word_, vacab_size_});
  aux_data_.resize(1);
  window_ = 0;
  //LOG(ERROR) << "row, col: " << row << ", " << col;
}

void DataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  string key, value;
  WordCharRecord ch;
  auto data = Tensor3(&data_);
  data = 0;
  //LOG(ERROR) << "Comp @ Data -----";
  if (store_ == nullptr) {
    store_ = singa::io::OpenStore(
        layer_conf_.GetExtension(data_conf).backend(),
        layer_conf_.GetExtension(data_conf).path(),
        singa::io::kRead);
    //store_->Read(&key, &value);
    //ch.ParseFromString(value);
    //SetInst(0, ch, &data_);
  }
  //ShiftInst(window_, 0, &data_);
  for (int batch = 0; batch < batchsize_; batch++) {
    int i = 0;
    for (int w = 0; w < max_num_word_; w++) {
      if (!store_->Read(&key, &value)) {
        store_->SeekToFirst();
        CHECK(store_->Read(&key, &value));
      }
      ch.ParseFromString(value);
      if (ch.word_index() == -1) {
        //window_ = i;      // the number of words in this sample
        //LOG(ERROR) << "window_: " << window_;
        break;
      }
      if (w == 0) {
        if (static_cast<int>(ch.label()) == 189)
          aux_data_.at(0) = 3;
        else
          aux_data_.at(0) = static_cast<int>(ch.label()) - 183;
      }
      for (int c = 0; c < max_word_len_; c++) {
        if (c >= ch.word_length())
          break;
        data[batch][i][ch.char_index(c)] = 1;
        i++;
      }
    }
    //LOG(ERROR) << "num of chars in batch" << batch << " " << i;
  }
}

/*******EmbeddingLayer**************/
EmbeddingLayer::~EmbeddingLayer() {
  delete embed_;
}

void EmbeddingLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Setup @ Embed";
  MSCNNLayer::Setup(conf, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  max_num_word_ = srclayers[0]->data(this).shape()[0];
  max_word_len_ = srclayers[0]->data(this).shape()[1] - 4;
  //LOG(ERROR) << "max_num_word_: " << max_num_word_;
  //LOG(ERROR) << "max_word_len_: " << max_word_len_;
  word_dim_ = conf.GetExtension(embedding_conf).word_dim();
  data_.Reshape(vector<int>{max_num_word_ * max_word_len_, word_dim_});
  grad_.ReshapeLike(data_);
  vocab_size_ = conf.GetExtension(embedding_conf).vocab_size();
  embed_ = Param::Create(conf.param(0));
  embed_->Setup(vector<int>{vocab_size_, word_dim_});
  //LOG(ERROR) << "data row, col: " << max_window << ", " << word_dim_;
  //LOG(ERROR) << "embd row, col: " << vocab_size_ << ", " << word_dim_;
}

void EmbeddingLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  //LOG(ERROR) << "Comp @ Embed";
  window_ = datalayer->window(); // <-- # of words in patient
  auto chars = RTensor2(&data_);
  auto embed = RTensor2(embed_->mutable_data());
  chars = 0;

  const float* idxptr = datalayer->data(this).cpu_data();
  //int i = 0;
  int shift = datalayer->data(this).shape()[1];
  //LOG(ERROR) << "max_word_len_: " << max_word_len_;
  for (int k=0; k < window_; k++) {
    int wlen = static_cast<int>(idxptr[k*shift+2]);
    //LOG(ERROR) << "wlen: " << wlen;
    for(int c=0; c<wlen; c++) {
      int char_idx = static_cast<int>(idxptr[k*shift+4+c]);  // "4": start position of char index
      //LOG(ERROR) << "char index: " << char_idx;
      CHECK_GE(char_idx, 0);
      CHECK_LT(char_idx, vocab_size_);
      Copy(chars[k * max_word_len_ + c], embed[char_idx]);
      //for (int i = 0; i < word_dim_; i++)
      //  chars[k * max_word_len_ + c][i] = embed[char_idx][i];
    }
    // show max @ embedding
    /*for (int i = 0; i < word_dim_; i++) {
      int max = k*max_word_len_;
      for (int c = 0; c < wlen; c++) {
        //if (k*max_word_len_+c == 138)
        //  LOG(ERROR) << "138 @ embedding: " << chars[k*max_word_len_+c][i];
        if (chars[k*max_word_len_+c][i] > chars[max][i]) {
          max = k*max_word_len_+c;
        }
      }
      LOG(ERROR) << "max " << i << " @ embedding: " << max << " " << chars[max][i];
    }*/
  }
  // show embedding vector
  //for (int j = 0; j < word_dim_; j++)
  //  LOG(ERROR) <<  "542, " << j << " " << embed[542][j];
}

void EmbeddingLayer::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "ComputeGradient @ embedding---";
  auto grad = RTensor2(&grad_);
  auto gembed = RTensor2(embed_->mutable_grad());
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[0]);
  gembed = 0;
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int k = 0; k < window_; k++) {
    int wlen = static_cast<int>(idxptr[k*shift+2]);
    for(int c=0; c<wlen; c++)
    {
      int char_idx = static_cast<int>(idxptr[k*shift+4+c]);  // "4": start position of char index
      //Copy(gembed[char_idx], grad[i++]);
      //LOG(ERROR) << "char_idx: " << char_idx;
      for (int t = 0; t < word_dim_; t++) {
        gembed[char_idx][t] += grad[k * max_word_len_ + c][t];
        //LOG(ERROR) << k*max_word_len_+c << " " << t << ": " << grad[k * max_word_len_ + c][t];
        //LOG(ERROR) << char_idx << " " << t << ": " << gembed[char_idx][t];
      }
      //if (char_idx = 542)
      //  for (int t = 0; t < word_dim_; t++)
      //    LOG(ERROR) << "grad @ 542: " << gembed[542][t];
    }
  }
}

/*************Concat Layer***************/
int ConcatLayer::Binomial(int n, int k) {
  if (k < 0 || k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  int cb = 1;
  for (int i = 1; i <= k; i++) {
    cb *= (n - (k -i));
    cb /= i;
  }
  return cb;
}

void ConcatLayer::Combinations(int n, int k) {
  int c = Binomial(n, k);
  int *ind =new int[k];
  for (int i = 0; i < k; i++)
    ind[i] = i;
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < k; j++)
      concat_index_[i][j] = ind[j];
    int x = k - 1;
    bool loop;
    do {
      loop = false;
      ind[x] = ind[x] + 1;
      if (ind[x] > n - (k - x)) {
        x--;
        loop = (x >= 0);
      }
      else {
        for (int x1 = x + 1; x1 < k; x1++)
          ind[x1] = ind[x1 - 1] + 1;
      }
    } while(loop);
  }
  delete[] ind;
}

void ConcatLayer::SetIndex(const vector<Layer*>& srclayers) {
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();    // #words in a patient
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int i = 0; i < window_; i++) {
    int wlen = static_cast<int>(idxptr[i * shift + 2]);
    word_index_[i] = wlen;
    /*for (int j = 0; j < wlen; j++) {
      int c = static_cast<int>(idxptr[i * shift + 4 + j]);
      char_index_[i][j] = c;
    }*/
  }
}

ConcatLayer::~ConcatLayer() {
  if (word_index_ != nullptr)
      delete[] word_index_;
  /*if (char_index_ != nullptr) {
    for (int i = 0; i < max_num_word_; i++)
      delete[] char_index_[i];
    delete[] char_index_;
  }*/
  if (concat_index_ != nullptr) {
    int b = Binomial(max_word_len_, kernel_);
    for (int i = 0; i < b; i++)
      delete[] concat_index_[i];
    delete[] concat_index_;
  }
}

void ConcatLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  Layer::Setup(conf, srclayers);
  //LOG(ERROR) << "Setup @ Concat";
  CHECK_EQ(srclayers.size(), 2);
  max_num_word_ = srclayers[1]->data(this).shape()[0];
  max_word_len_ = srclayers[1]->data(this).shape()[1] - 4;
  word_dim_ = srclayers[0]->data(this).count() /
    srclayers[0]->data(this).shape()[0];
  //max_word_len_ = conf.GetExtension(data_conf).max_word_len();
  //max_num_word_ = conf.GetExtension(data_conf).max_num_word();
  //word_dim_ = conf.GetExtension(embedding_conf).word_dim();
  kernel_ = conf.GetExtension(concat_conf).kernel();
  //LOG(ERROR) << "kernel: " << kernel_;
  //LOG(ERROR) << "word_dim_: " << word_dim_;
  //LOG(ERROR) << "max_num_word_: " << max_num_word_;
  //LOG(ERROR) << "max_word_len_: " << max_word_len_;
  int cols = kernel_ * word_dim_;
  int bino = Binomial(max_word_len_, kernel_);
  //LOG(ERROR) << "bino: " << bino;
  int rows = max_num_word_ * bino;
  //LOG(ERROR) << "rows: " << rows;
  //LOG(ERROR) << "cols: " << cols;
  data_.Reshape(vector<int>{rows, cols});
  grad_.ReshapeLike(data_);
  // length of each word
  word_index_ = new int[max_num_word_];
  // character index of the word
  //char_index_ = new int*[max_num_word_];
  //for (int i = 0; i < max_num_word_; i++)
  //  char_index_[i] = new int[max_word_len_];
  // for each word, a concat_index_ stores all the combinations
  concat_index_ = new int*[bino];
  for (int i = 0; i < bino; i++)
    concat_index_[i] = new int[kernel_];
}

void ConcatLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "CompFeature @ Concat";
  auto emlayer = dynamic_cast<EmbeddingLayer*>(srclayers[0]);
  window_ = emlayer->window();      // #words in a patient
  auto data = Tensor2(&data_);
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  data = 0;
  SetIndex(srclayers);

  int max = Binomial(max_word_len_, kernel_);
  for (int w = 0; w < window_; w++) {
    Combinations(word_index_[w], kernel_);
    int b = Binomial(word_index_[w], kernel_);
    //LOG(ERROR) << "b: " << b << "----------";
    for (int r = 0; r < b; r++) {
      for (int i = 0; i < kernel_ * word_dim_; i++) {
        data[w * max + r][i] = src[w * max_word_len_ + concat_index_[r][i / word_dim_]][i % word_dim_];
      }
    }
    /*int t = w * max;
    for (int i = 0; i < kernel_ * word_dim_; i++) {
      for (int r = 0; r < b; r++) {
        if (data[w * max + r][i] > data[t][i]) {
          t = w*max+r;
        }
      }
      LOG(ERROR) << "max @ concat: " << t << " " << data[t][i];
    }*/
  }


  /*auto src_ptr = emlayer->data(this).cpu_data();
  float* dst_ptr = data_.mutable_cpu_data();
  data_.SetValue(0);
  SetIndex(srclayers);

  int max = Binomial(max_word_len_, kernel_);
  //LOG(ERROR) << "start concatenation";
  for (int w = 0; w < window_; w++) {
    //LOG(ERROR) << "word_index_[" << w << "]: " << word_index_[w];
    //LOG(ERROR) << "kernel_: " << kernel_;
    Combinations(word_index_[w], kernel_);
    //LOG(ERROR) << "set concat_index_ ok!";
    int b = Binomial(word_index_[w], kernel_);
    LOG(ERROR) << "b: " << b;
    for (int r = 0; r < b; r++)
      for (int c = 0; c < kernel_; c++) {
        memcpy(dst_ptr + ((w * max + r) * kernel_ + c) * word_dim_,
        src_ptr + (w * max_word_len_ + concat_index_[r][c]) * word_dim_, word_dim_);
        LOG(ERROR) << "r, " << r << "c" << c << " @ concat: " << concat_index_[r][c];
        for (int i = 0; i < word_dim_; i++)
          LOG(ERROR) << i << " " << *(src_ptr + (w * max_word_len_ + concat_index_[r][c]) * word_dim_ + i);
        for (int i = 0; i < word_dim_; i++)
          LOG(ERROR) << i << " " << *(dst_ptr + ((w * max + r) * kernel_ + c) * word_dim_ + i);
      }
    // show max
    for (int i = 0; i < kernel_*word_dim_; i++) {
      LOG(ERROR) << "0, " << i << " @ concat: " << *(dst_ptr+i);
      int m = w * max_word_len_;
      for (int c = 0; c < b; c++) {
        if (*(dst_ptr+(w*max+c)*kernel_*word_dim_+i) >
            *(dst_ptr+m*kernel_*word_dim_+i)) {
          m = w*max_word_len_+c;
        }
      }
      LOG(ERROR) << "max " << i << " @ concat: " << m << " " << *(dst_ptr+m*kernel_*word_dim_+i);
    }
  }*/
  //LOG(ERROR) << "end concatenation";
}

void ConcatLayer::ComputeGradient(int flag, const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "ComputeGradient @ Concat";
  auto grad = Tensor2(&grad_);
  auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
  auto emlayer = dynamic_cast<EmbeddingLayer*>(srclayers[0]);
  window_ = emlayer->window();      // #words in a patient
  gsrc = 0;

  int max = Binomial(max_word_len_, kernel_);
  for (int w = 0; w < window_; w++) {
    Combinations(word_index_[w], kernel_);
    //LOG(ERROR) << "w: " << w << "----------";
    int b = Binomial(word_index_[w], kernel_);
    for (int r = 0; r < b; r++)
      for (int c = 0; c < kernel_; c++)
        for (int i = 0; i < word_dim_; i++) {
          gsrc[w * max_word_len_ + concat_index_[r][c]][i]
            += grad[w * max + r][c * word_dim_ + i];
          //LOG(ERROR) << w*max+r << " " << c*word_dim_+i << ": " << grad[w * max + r][c * word_dim_ + i];
        }
  }
}

/*********PoolingOverTime Layer*********/
void PoolingOverTime::SetIndex(const vector<Layer*>& srclayers) {
  // pay attention to the index of data layer
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();    // #words in a patient
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int i = 0; i < window_; i++) {
    int wlen = static_cast<int>(idxptr[i * shift + 2]);
    word_index_[i] = wlen;
  }
}

int PoolingOverTime::Binomial(int n, int k) {
  if (k < 0 || k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  int cb = 1;
  for (int i = 1; i <= k; i++) {
    cb *= (n - (k -i));
    cb /= i;
  }
  return cb;
}

PoolingOverTime::~PoolingOverTime() {
  if (word_index_ != nullptr)
    delete[] word_index_;
  if (max_index_ != nullptr) {
    for (int i = 0; i < max_num_word_; i++)
      delete[] max_index_[i];
    delete[] max_index_;
  }
}

void PoolingOverTime::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Setup @ PoolingOverTime";
  CHECK_EQ(srclayers.size(), 2);
  MSCNNLayer::Setup(conf, srclayers);

  kernel_ = conf.GetExtension(pot_conf).kernel();
  //LOG(ERROR) << "kernel_: " << kernel_;
  max_num_word_ = srclayers[1]->data(this).shape()[0];
  max_word_len_ = srclayers[1]->data(this).shape()[1] - 4;
  //LOG(ERROR) << "max_word_len_: " << max_word_len_;
  //LOG(ERROR) << "max_num_word_: " << max_num_word_;

  //int conv_dim = srclayers[0]->data(this).shape().size();
  //batchsize_ = srclayers[0]->data(this).shape()[conv_dim - 2];
  //vdim_ = srclayers[0]->data(this).shape()[conv_dim - 3];
  batchsize_ = srclayers[0]->data(this).shape()[0];
  vdim_ = srclayers[0]->data(this).count() / batchsize_;
  //LOG(ERROR) << "batchsize_: " << batchsize_;
  //LOG(ERROR) << "vdim_: " << vdim_;

  max_row_ = batchsize_ / max_num_word_;
  //LOG(ERROR) << "max_row_: " << max_row_;
  // will append time from data layer
  data_.Reshape(vector<int>{1, max_num_word_, vdim_ + 1});
  grad_.ReshapeLike(data_);
  word_index_ = new int[max_num_word_];
  // indicate the index of the maximum element
  max_index_ = new int*[max_num_word_];
  for (int i = 0; i < max_num_word_; i++)
    max_index_[i] = new int[vdim_];
}

void PoolingOverTime::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "CompFeat @ PoolingOverTime";
  auto data = Tensor3(&data_);
  auto src = Tensor2(srclayers[0]->mutable_data(this));
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window();    // #words in a patient
  //auto concatlayer = dynamic_cast<ConcatLayer*>(srclayers[2]);
  //int kernel = concatlayer->kernel();
  SetIndex(srclayers);
  //int max = Binomial(max_word_len_, kernel_);
  for (int w = 0; w < window_; w++) {
    for (int c = 0; c < vdim_; c++) {
      data[0][w][c] = src[w * max_row_][c];
      max_index_[w][c] = w * max_row_;
    }
    int b = Binomial(word_index_[w], kernel_);
    //LOG(ERROR) << "b: " << b;
    for (int c = 0; c < vdim_; c++)
      for (int r = 1; r < b; r++) {
        /*if (w * max_row_ + r == 138)
          LOG(ERROR) << "138, " << c << "@PoolingOverTime: " << src[w*max_row_+r][c];*/
        if (src[w * max_row_ + r][c] > data[0][w][c]) {
          data[0][w][c] = src[w * max_row_ + r][c];
          max_index_[w][c] = w * max_row_ + r;
        }
      }
    /*for (int c = 0; c < vdim_; c++) {
      LOG(ERROR) << "c" << c << ": " << max_index_[w][c] << " " << src[max_index_[w][c]][c];
      //LOG(ERROR) << "139, " << c << "@PoolingOverTime: " << src[139][c];
    }*/
  }
  // append time from data layer
  const float* idxptr = datalayer->data(this).cpu_data();
  int shift = datalayer->data(this).shape()[1];
  for (int i = 0; i < window_; i++) {
    int delta_time = static_cast<int>(idxptr[i * shift + 3]);
    data[0][i][vdim_] = delta_time;
  }
}

void PoolingOverTime::ComputeGradient(int flag,
    const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "ComputeGradient @ PoolingOverTime";
  auto grad = Tensor3(&grad_);
  //srclayers[0]->mutable_grad(this)->SetValue(0);
  auto gsrc = Tensor2(srclayers[0]->mutable_grad(this));
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  gsrc = 0;
  window_ = datalayer->window();
  for (int w = 0; w < window_; w++) {
    //LOG(ERROR) << "w: " << w << "---------";
    for (int c = 0; c < vdim_; c++) {
      gsrc[max_index_[w][c]][c] = grad[0][w][c];
      //LOG(ERROR) << "grad @ PoolingOverTime (" << max_index_[w][c] << " " << c << "): " << gsrc[max_index_[w][c]][c];
    }
  }
}


/*******WordPoolingLayer*********/
WordPoolingLayer::~WordPoolingLayer() {
  if (index_ != nullptr)
    delete[] index_;
}

void WordPoolingLayer::Setup(const LayerProto& conf,
  const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "Setup @ WordPoolingLayer";
  CHECK_EQ(srclayers.size(), 2);
  MSCNNLayer::Setup(conf, srclayers);
  const auto& src = srclayers[0]->data(this);
  int dim = src.shape().size();
  batchsize_ = src.shape()[dim - 2];
  vdim_ = src.shape()[dim - 3];
  data_.Reshape(vector<int>{1, vdim_});
  grad_.ReshapeLike(data_);
  index_ = new int[vdim_];
}

void WordPoolingLayer::ComputeFeature(int flag,
  const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "ComputeFeature @ WordPoolingLayer";
  auto datalayer = dynamic_cast<DataLayer*>(srclayers[1]);
  window_ = datalayer->window(); // <-- # of words in patient
  auto data = Tensor2(&data_);
  auto src = Tensor4(srclayers[0]->mutable_data(this));
  for (int i = 0; i < vdim_; i++) {
    data[0][i] = src[0][i][0][0];
    index_[i] = 0;
  }
  //LOG(ERROR) << "batchsize_: " << batchsize_;
  for (int i = 0; i < vdim_; i++)
    for (int j = 1; j < window_; j++)
      if (data[0][i] < src[0][i][j][0]) {
        data[0][i] = src[0][i][j][0];
        index_[i] = j;
      }
  /*for (int i = 0; i < vdim_; i++) {
    LOG(ERROR) << "max @ WordPoolingLayer: " << index_[i] << " " << src[0][i][index_[i]][0];
  }*/
}

void WordPoolingLayer::ComputeGradient(int flag,
  const vector<Layer*>& srclayers) {
  //LOG(ERROR) << "ComputeGradient @ WordPoolingLayer";
  auto grad = Tensor2(&grad_);
  //srclayers[0]->mutable_grad(this)->SetValue(0);
  auto gsrc = Tensor4(srclayers[0]->mutable_grad(this));
  gsrc = 0;
  for (int i = 0; i < vdim_; i++) {
    gsrc[0][i][index_[i]][0] = grad[0][i];
    //LOG(ERROR) << "grad from WordPoolingLayer to convnet (" << i << " " << index_[i] << "): " << gsrc[0][i][index_[i]][0];
  }
}


}   // end of namespace mscnnlm
