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

#ifndef EXAMPLES_MSCNNLM_MSCNNLM_H_
#define EXAMPLES_MSCNNLM_MSCNNLM_H_

#include <string>
#include <vector>
#include "singa/singa.h"
#include "./mscnnlm.pb.h"

namespace mscnnlm {
using std::vector;
using singa::LayerProto;
using singa::Layer;
using singa::Param;
using singa::Blob;
using singa::Metric;

using namespace mshadow;
using mshadow::Tensor;
/**
 * Base RNN layer. May make it a base layer of SINGA.
 */
class MSCNNLayer : virtual public singa::Layer {
 public:
  /**
   * The recurrent layers may be unrolled different times for different
   * iterations, depending on the applications. For example, the ending word
   * of a sentence may stop the unrolling; unrolling also stops when the max
   * window size is reached. Every layer must reset window_ in its
   * ComputeFeature function.
   *
   * @return the effective BPTT length, which is <= max_window.
   */
  inline int window() { return window_; }

 protected:
  //!< effect window size for BPTT
  int window_;
};

/**
 * Input layer that get read records from data shard
 */
class DataLayer : public MSCNNLayer, public singa::InputLayer {
 public:
  ~DataLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  int max_window() const {
    return max_window_;
  }

 private:
  int num_feature_;
  int max_window_;
  int max_word_len_;
  int max_num_word_;
  singa::io::Store* store_ = nullptr;
};


/**
 * LabelLayer that read records_[1] to records_[window_] from DataLayer to
 * offer label information
class LabelLayer : public RNNLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
};
 */


/**
 * Word embedding layer that get one row from the embedding matrix for each
 * word based on the word index
 */
class EmbeddingLayer : public MSCNNLayer {
 public:
  ~EmbeddingLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{embed_};
    return params;
  }


 private:
  int word_dim_;
  int vocab_size_;
  int max_num_word_;
  int max_word_len_;
  //!< word embedding matrix of size vocab_size_ x word_dim_
  Param* embed_;
};

class ConcatLayer : public MSCNNLayer {
 public:
  ~ConcatLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  int Binomial(int n, int k);
  void Combinations(int n, int k);
  void SetIndex(const vector<Layer*>& srclayers);
 private:
  int max_word_len_;
  int max_num_word_;
  int word_dim_;
  int kernel_;
  int *word_index_;       // len of each word
  //int **char_index_;      // character index in each word
  int **concat_index_;    // all k subsets of n elementes
};

class PoolingOverTime : public MSCNNLayer {
 public:
  ~PoolingOverTime();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
  void SetIndex(const vector<Layer*>& srclayers);
  int Binomial(int n, int k);
 private:
  int batchsize_;
  int vdim_;
  int max_num_word_;
  int max_word_len_;
  int max_row_;
  //int *word_index_;
  int **max_index_;
};

class WordPoolingLayer: public MSCNNLayer {
 public:
  ~WordPoolingLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
 private:
   int batchsize_;
   int vdim_;
   int *index_;
};

}  // namespace mscnnlm
#endif  // EXAMPLES_MSCNNLM_MSCNNLM_H_
