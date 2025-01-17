/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "TensorManager.h"

#include <util/logging.h>

namespace
{

using namespace onert;

template <typename Tensor>
void allocateMemory(backend::train::MemoryManager *mgr,
                    const ir::OperandIndexMap<std::unique_ptr<Tensor>> &tensors,
                    const std::string tensor_type)
{
  mgr->allocate();

  for (auto &&pair : tensors)
  {
    const auto &index = pair.first;
    auto tensor = pair.second.get();
    assert(!tensor->is_dynamic());

    auto *buffer = mgr->getBuffer(index);
    tensor->setBuffer(buffer);
    VERBOSE(TensorManager) << tensor_type << index << " : " << static_cast<void *>(buffer)
                           << std::endl;
  }
}

} // namespace

namespace onert
{
namespace backend
{
namespace train
{

TensorManager::TensorManager(const std::shared_ptr<TensorRegistry> &reg,
                             const std::string planner_id)
  : _nonconst_mgr{new MemoryManager(planner_id)}, _trainable_mgr{new MemoryManager(planner_id)},
    _derivative_mgr{new MemoryManager(planner_id)},
    _gradient_mgr{new MemoryManager(planner_id)}, _tensors{reg}
{
  // DO NOTHING
}

void TensorManager::allocateNonConstTensors()
{
  allocateMemory(_nonconst_mgr.get(), _tensors->nonconst_tensors(),
                 std::string{"          TENSOR "});
}

void TensorManager::allocateTrainableTensors()
{
  allocateMemory(_trainable_mgr.get(), _tensors->trainable_tensors(),
                 std::string{"TRAINABLE TENSOR "});
}

void TensorManager::allocateDerivativeTensors()
{
  allocateMemory(_derivative_mgr.get(), _tensors->derivative_tensors(),
                 std::string{"DERIVATIVE TENSOR "});
}

void TensorManager::allocateGradientTensors()
{
  allocateMemory(_gradient_mgr.get(), _tensors->gradient_tensors(),
                 std::string{"GRADIENT TENSOR "});
}

void TensorManager::claimNonConstPlan(const ir::OperandIndex &index)
{
  auto tensor = _tensors->getNonConstTensor(index);
  assert(tensor && !tensor->is_dynamic());

  auto size = tensor->total_size();
  _nonconst_mgr->claimPlan(index, size);
}

void TensorManager::releaseNonConstPlan(const ir::OperandIndex &index)
{
  assert(_tensors->getNonConstTensor(index) && !_tensors->getNonConstTensor(index)->is_dynamic());

  _nonconst_mgr->releasePlan(index);
}

void TensorManager::claimTrainablePlan(const ir::OperandIndex &index)
{
  auto tensor = _tensors->getTrainableTensor(index);
  assert(tensor && !tensor->is_dynamic());

  auto size = tensor->total_size();
  _trainable_mgr->claimPlan(index, size);
}

void TensorManager::releaseTrainablePlan(const ir::OperandIndex &index)
{
  assert(_tensors->getTrainableTensor(index) && !_tensors->getTrainableTensor(index)->is_dynamic());

  _trainable_mgr->releasePlan(index);
}

void TensorManager::claimDerivativePlan(const ir::OperandIndex &index)
{
  auto tensor = _tensors->getDerivativeTensor(index);
  assert(tensor && !tensor->is_dynamic());

  auto size = tensor->total_size();
  _derivative_mgr->claimPlan(index, size);
}

void TensorManager::releaseDerivativePlan(const ir::OperandIndex &index)
{
  assert(_tensors->getDerivativeTensor(index) &&
         !_tensors->getDerivativeTensor(index)->is_dynamic());

  _derivative_mgr->releasePlan(index);
}

void TensorManager::claimGradientPlan(const ir::OperandIndex &index)
{
  auto tensor = _tensors->getGradientTensor(index);
  assert(tensor && !tensor->is_dynamic());

  auto size = tensor->total_size();
  _gradient_mgr->claimPlan(index, size);
}

void TensorManager::releaseGradientPlan(const ir::OperandIndex &index)
{
  assert(_tensors->getGradientTensor(index) && !_tensors->getGradientTensor(index)->is_dynamic());

  _gradient_mgr->releasePlan(index);
}

} // namespace train
} // namespace backend
} // namespace onert
