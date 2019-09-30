// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/cpu/ml/linearregressor.h"

#include "core/util/math.h"

namespace onnxruntime {
namespace ml {

ONNX_CPU_OPERATOR_ML_KERNEL(LinearRegressor, 1,
                            KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                            LinearRegressor<float>);

template <typename T>
LinearRegressor<T>::LinearRegressor(const OpKernelInfo& info)
    : OpKernel(info),
      intercepts_(info.GetAttrsOrDefault<float>("intercepts")),
      post_transform_(MakeTransform(info.GetAttrOrDefault<std::string>("post_transform", "NONE"))) {
  ORT_ENFORCE(info.GetAttr<int64_t>("targets", &targets_).IsOK());
  ORT_ENFORCE(info.GetAttrs<float>("coefficients", coefficients_).IsOK());
}

template <>
Status LinearRegressor<float>::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  concurrency::ThreadPool* tp = ctx_internal->GetOperatorThreadPool();

  const auto* X = ctx->Input<Tensor>(0);
  if (X->Shape().NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }
  // X: [N, feature_size],coefficients_[target, feature_size]
  // X*coefficients_^t : [N, target]
  int64_t stride = X->Shape().NumDimensions() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().NumDimensions() == 1 ? 1 : X->Shape()[0];
  Tensor* Y = ctx->Output(0, TensorShape({N, targets_}));
  float* Ydata = Y->MutableData<float>();
  // TODO: if N == 1 or stride == 1 or targets_== 1, we can use GEMV
  math::Gemm<float>(CblasNoTrans, CblasTrans, N, targets_, stride, 1.0f, X->Data<float>(), coefficients_.data(), 0.0f,
                    Y->MutableData<float>(), tp);
  bool useIntercepts = intercepts_.size() == static_cast<size_t>(targets_);
  if (useIntercepts) {
    for (int64_t i = 0; i < N; i++)  // for each point
    {
      float* p = Ydata + i * targets_;
      for (int j = 0; j < targets_; j++)  // for each target
      {
        p[j] += intercepts_[j];
      }
    }
  }
  // TODO: parallel this part
  if (post_transform_ != POST_EVAL_TRANSFORM::NONE)
    for (int64_t i = 0; i < N; i++)  // for each point
    {
      float* p = Ydata + i * targets_;
      ml::write_scores(p, targets_, post_transform_, p);
    }
  return Status::OK();
}

}  // namespace ml
}  // namespace onnxruntime
