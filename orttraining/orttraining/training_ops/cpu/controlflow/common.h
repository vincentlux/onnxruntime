// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {

template <int input_start, int output_start>
std::vector<std::pair<int, int>> AliasRange(int start, int end) {
  std::vector<std::pair<int, int>> aliases;
  for (int i = start; i < end; i++) {
    aliases.emplace_back(std::pair<int, int>(input_start + i, output_start + i));
  }
  return aliases;
}

inline void ComputeTensorSizeAndBufferLength(OpKernelContext* context,
                                             std::vector<int>& tensor_element_counts,
                                             std::vector<size_t>& tensor_offsets,
                                             std::vector<size_t>& tensor_sizes,
                                             int64_t& total_buffer_len) {
  size_t size_in_bytes = 0;
  const int num_tensors = context->InputCount();
  for (int i = 0; i < num_tensors; ++i) {
    const Tensor* x_tensor = context->Input<Tensor>(i);
    tensor_offsets.push_back(size_in_bytes);

    size_in_bytes = x_tensor->SizeInBytes();
    total_buffer_len += size_in_bytes;

    tensor_sizes.push_back(size_in_bytes);
    tensor_element_counts.push_back((int)x_tensor->Shape().Size());
  }
}
}  // namespace contrib
}  // namespace onnxruntime
