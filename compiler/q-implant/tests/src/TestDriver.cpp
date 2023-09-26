/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <luci/ImporterEx.h>
#include <luci/CircleQuantizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>
#include <luci/IR/CircleNode.h>
#include <loco.h>

#include <npy.hpp>
#include <json.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>

#define THROW_UNLESS(cond) \
  if (not(cond))           \
    throw std::runtime_error{#cond};

namespace
{
// Return directory path of given file path
// TODO Find a platform-independent way to do this
std::string directory_path(const std::string &file_path)
{
  const auto pos = file_path.find_last_of("/");
  if (std::string::npos == pos)
    return "";

  return file_path.substr(0, pos);
}

loco::DataType str_to_dtype(const std::string &str)
{
  auto lower_case_str = str;
  std::transform(lower_case_str.begin(), lower_case_str.end(), lower_case_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_case_str.compare("int8") == 0)
    return loco::DataType::S8;
  if (lower_case_str.compare("uint8") == 0)
    return loco::DataType::U8;
  if (lower_case_str.compare("int16") == 0)
    return loco::DataType::S16;
  if (lower_case_str.compare("int32") == 0)
    return loco::DataType::S32;
  if (lower_case_str.compare("int64") == 0)
    return loco::DataType::S64;

  throw std::runtime_error("Invalid dtype detected. " + str);
}

// Throw an exception if tensor has any invalid field.
void verify_tensor(const Json::Value &tensor)
{
  THROW_UNLESS(tensor.isMember("scale"));
  THROW_UNLESS(tensor["scale"].isString());
  THROW_UNLESS(tensor.isMember("zerop"));
  THROW_UNLESS(tensor["zerop"].isString());
  THROW_UNLESS(tensor.isMember("quantized_dimension"));
  THROW_UNLESS(tensor["quantized_dimension"].isUInt());
  THROW_UNLESS(tensor.isMember("dtype"));
  THROW_UNLESS(tensor["dtype"].isString());

  if (tensor.isMember("value"))
  {
    THROW_UNLESS(tensor["value"].isString());
  }
}

Json::Value load_json(const std::string &path)
{
  Json::Value root;
  std::ifstream ifs(path);

  // Failed to open cfg file
  if (not ifs.is_open())
    throw std::runtime_error("Cannot open config file. " + path);

  Json::CharReaderBuilder builder;
  JSONCPP_STRING errs;

  // Failed to parse
  if (not parseFromStream(builder, ifs, &root, &errs))
    throw std::runtime_error("Cannot parse config file (json format). " + errs);

  return root;
}

bool check_dtype(loco::Graph *g)
{
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);

    // Throw an exception if dtype is not float32
    // TODO Operator-level verification (ex: using QuantizedModelVerifier)
    if (circle_node->dtype() == loco::DataType::FLOAT32)
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

bool check_scale(luci::CircleNode *input_node, luci::CircleNode *output_node,
                 const std::string &scale_path)
{
  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<float> scale;
  npy::LoadArrayFromNumpy(scale_path, shape, fortran_order, scale);

  THROW_UNLESS(shape.size() == 1);
  THROW_UNLESS(output_node->quantparam()->scale.size() == shape[0]);
  THROW_UNLESS(fortran_order == false);

  for (int i = 0; i < shape[0]; ++i)
  {
    if (std::abs(output_node->quantparam()->scale[i] - scale[i]) > 1e-7)
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

bool check_zerop(luci::CircleNode *input_node, luci::CircleNode *output_node,
                 const std::string &zerop_path)
{
  std::vector<unsigned long> shape;
  bool fortran_order;
  std::vector<int64_t> zerop;
  npy::LoadArrayFromNumpy(zerop_path, shape, fortran_order, zerop);

  THROW_UNLESS(shape.size() == 1);
  THROW_UNLESS(output_node->quantparam()->zerop.size() == shape[0]);
  THROW_UNLESS(fortran_order == false);

  for (int i = 0; i < shape[0]; ++i)
  {
    if (output_node->quantparam()->zerop[i] != zerop[i])
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

bool check_value(loco::Graph *input, loco::Graph *output, const Json::Value &qparam,
                 const std::string &dir_path)
{
  std::unordered_map<std::string, luci::CircleNode *> map_input;
  std::unordered_map<std::string, luci::CircleNode *> map_output;

  for (auto node : loco::all_nodes(input))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
    {
      continue;
    }

    map_input[circle_node->name()] = circle_node;
  }
  for (auto node : loco::all_nodes(output))
  {
    auto circle_node = loco::must_cast<luci::CircleNode *>(node);
    if (circle_node->opcode() == luci::CircleOpcode::CIRCLEOUTPUT)
    {
      continue;
    }

    map_output[circle_node->name()] = circle_node;
  }
  for (const auto tensor_name : qparam.getMemberNames())
  {
    const auto tensor = qparam[tensor_name];

    verify_tensor(tensor);

    const auto scale_path = dir_path + '/' + tensor["scale"].asString();
    const auto zerop_path = dir_path + '/' + tensor["zerop"].asString();
    const auto quantized_dimension = tensor["quantized_dimension"].asUInt();
    const auto dtype = str_to_dtype(tensor["dtype"].asString());

    auto input_node = map_input.at(tensor_name);
    auto output_node = map_output.at(tensor_name);

    // check scale
    if (check_scale(input_node, output_node, scale_path))
    {
      std::cerr << "scale parameter of output is different from qparams!" << std::endl;
      return EXIT_FAILURE;
    }

    // check zerop
    if (check_zerop(input_node, output_node, zerop_path))
    {
      std::cerr << "zero point parameter of output is different from qparams!" << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

} // namespace

int entry(int argc, char **argv)
{
  // check arguments
  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0] << " <input.circle> <qparams> <output.circle>" << std::endl;
    return EXIT_FAILURE;
  }

  const auto input_path = std::string(argv[1]);
  const auto qparams_path = std::string(argv[2]);
  const auto output_path = std::string(argv[3]);

  luci::ImporterEx importerex;

  // Load input model from the file
  auto input_module = importerex.importVerifyModule(input_path);
  if (input_module.get() == nullptr)
    return EXIT_FAILURE;

  // open json
  const auto root = load_json(qparams_path);
  THROW_UNLESS(root.isObject());
  const auto dir_path = directory_path(qparams_path);

  // Load output model from the file
  auto output_module = importerex.importVerifyModule(output_path);
  if (output_module.get() == nullptr)
    return EXIT_FAILURE;

  if (input_module->size() != 1 || output_module->size() != 1)
  {
    std::cerr << "ERROR: Only a single subgraph is supported" << std::endl;
    return EXIT_FAILURE;
  }
  if (input_module->size() != output_module->size())
  {
    std::cerr << "ERROR: Subgraph sizes of input and output circle are different." << std::endl;
    return EXIT_FAILURE;
  }

  for (size_t idx = 0; idx < output_module->size(); ++idx)
  {
    auto input_graph = input_module->graph(idx);
    auto output_graph = output_module->graph(idx);

    if (check_dtype(output_graph))
    {
      std::cerr << "ERROR: Quantized tensor type is not int type!" << std::endl;
      return EXIT_FAILURE;
    }

    // TODO: check values of output graph
    if (check_value(input_graph, output_graph, root, dir_path))
    {
      std::cerr << "ERROR: Tensor value is wrong!" << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "[TEST PASSED]" << std::endl;
  return EXIT_SUCCESS;
}
