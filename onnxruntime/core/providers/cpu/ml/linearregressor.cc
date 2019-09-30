// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cpu/ml/linearregressor.h"

namespace onnxruntime {
namespace ml {

ONNX_OPERATOR_TYPED_KERNEL_EX(LinearRegressor, kMLDomain, 1, float, kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                              LinearRegressor<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(LinearRegressor, kMLDomain, 1, double, kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<double>()),
                              LinearRegressor<double>);

template <typename T>
LinearRegressor<T>::LinearRegressor(const OpKernelInfo& info)
    : OpKernel(info), post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  std::vector<float> c;
  c = info.GetAttrsOrDefault<float>("intercepts");
  intercepts_.resize(c.size());
  std::copy_n(c.data(), c.size(), intercepts_.data());
  ORT_ENFORCE(info.GetAttr<int64_t>("targets", &targets_).IsOK());
  c.clear();
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", c).IsOK());
  coefficients_.resize(c.size());
  std::copy_n(c.data(), c.size(), coefficients_.data());
}

}  // namespace ml
}  // namespace onnxruntime
