#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/allclose.h>
#include <ATen/ops/rand.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torchvision/vision.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "passes/fuse_ops.h"
#include "passes/te_op.h"
#include "util/common.h"
#include "util/logging.h"

using json = nlohmann::json;
using namespace torch::jit;

static Value *createValue(TypePtr type, const json &input, Graph *graph) {
  switch (type->kind()) {
    case TypeKind::IntType: {
      return graph->insertConstant(input.get<int64_t>());
    } break;

    case TypeKind::BoolType: {
      return graph->insertConstant(input.get<bool>());
    } break;

    case TypeKind::FloatType: {
      return graph->insertConstant(input.get<float>());
    } break;

    case TypeKind::TensorType: {
      auto shape = input.at("shape").get<std::vector<int64_t>>();
      return graph->addInput()->setType(
          TensorType::createContiguous(c10::kFloat, c10::kCUDA, shape));
    } break;

    case TypeKind::ListType: {
      auto elemType = type->cast<ListType>()->getElementType();
      auto elemJsons = input.get<std::vector<json>>();
      std::vector<Value *> elemValues;
      for (auto &elem : elemJsons)
        elemValues.push_back(createValue(elemType, elem, graph));
      auto list = graph->appendNode(graph->createList(elemType, elemValues));
      return list->output(0);
    } break;

    default: {
      TORCH_CHECK(false, "Type ", *type, " not supported.");
    }
  }
}

static Value *createNode(const json &inputCase, const FunctionSchema &schema,
                         Graph *graph) {
  // Get symbol
  auto symbol = Symbol::fromQualString(schema.name());

  // Parse positional arguments
  auto argJsons = inputCase.at(0).get<std::vector<json>>();
  std::vector<NamedValue> argValues;
  for (auto i : c10::irange(argJsons.size())) {
    auto &input = argJsons[i];
    auto type = schema.arguments()[i].type();
    argValues.push_back(createValue(type, input, graph));
  }

  // Parse keyword arguments
  auto kwargJsons =
      inputCase.at(1).get<std::unordered_map<std::string, json>>();
  std::vector<NamedValue> kwargValues;
  for (auto &pair : kwargJsons) {
    auto argIdx = *schema.argumentIndexWithName(pair.first);
    auto type = schema.arguments()[argIdx].type();
    kwargValues.emplace_back(pair.first, createValue(type, pair.second, graph));
  }

  return graph->insert(symbol, argValues, kwargValues);
}

static void createFusedFunctor(const std::shared_ptr<Graph> &graph) {
  // Infer dtype and device
  ValueTypeMap refinedTypes;
  InferDtypeAndDevice(graph, refinedTypes);

  // Include `ListConstruct` in the graph
  auto tail = graph->return_node(), head = tail->prev();
  for (auto node = head->prev(); node != graph->param_node();
       node = node->prev()) {
    if (node->kind() == prim::ListConstruct) {
      auto tmp = node->next();
      node->moveBefore(head);
      head = node;
      node = tmp;
    }
  }

  // Create fusion group
  commitFusion(head, tail, graph.get(), refinedTypes);

  // Create fusion functor
  FusedOpToParallization(graph, refinedTypes);
  MapFunctorToParallization(graph, refinedTypes);
}

static auto rng = at::make_generator<at::CUDAGeneratorImpl>();

static IValue generateInput(TypePtr type) {
  switch (type->kind()) {
    case TypeKind::TensorType: {
      auto tensorTy = type->cast<TensorType>();
      return at::rand(*tensorTy->sizes().concrete_sizes(), rng,
                      tensorTy->scalarType(), c10::kStrided, tensorTy->device(),
                      c10::nullopt);
    } break;

    default: {
      TORCH_CHECK(false, "Cannot generate input for type ", *type);
    }
  }
}

static void runCase(const json &inputCase, const FunctionSchema &schema) {
  // Construct reference graph
  auto refGraph = std::make_shared<Graph>();
  refGraph->registerOutput(createNode(inputCase, schema, refGraph.get()));
  ConstantPropagation(refGraph);
  refGraph->dump();

  // Construct graph with fused functor
  auto compiledGraph = refGraph->copy();
  createFusedFunctor(compiledGraph);
  compiledGraph->dump();

  // Generate inputs
  std::vector<IValue> inputs;
  for (auto value : refGraph->inputs())
    inputs.push_back(generateInput(value->type()));

  // Run reference graph
  at::Tensor refOut;
  {
    auto stack = inputs;
    GraphFunction func("test", refGraph, nullptr);
    func.run(stack);
    refOut = stack.front().toTensor();
  }

  // Run compiled graph
  at::Tensor compileOut;
  {
    Code code(compiledGraph, "test");
    auto stack = inputs;
    InterpreterState(code).run(stack);
    compileOut = stack.front().toTensor();
  }
  // Compare result
  TORCH_CHECK(at::allclose(refOut, compileOut, 1e-3, 1e-5));
}

static void runOpSuite(const json &opSuite) {
  // Find operator from schema
  auto schema = parseSchema(opSuite.at("schema"));
  auto opName = schema.operator_name();
  auto op = findOperatorFor(schema.operator_name());
  TORCH_CHECK(op, "Operator not found for ", opName);

  // Run each test case
  auto inputCases = opSuite.at("inputs").get<std::vector<json>>();
  for (auto &testCase : inputCases) runCase(testCase, schema);
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: test_lowering <test-suite-json> <suite-name>...";
    return 1;
  }

  // Load test suite from file
  at::globalContext().lazyInitCUDA();
  std::ifstream suiteFile(argv[1]);
  auto suite = json::parse(suiteFile);

  // Run test suite
  for (auto i = 2u; i < argc; i++) {
    auto opSuite = suite.at(argv[i]);
    runOpSuite(opSuite);
  }

  return 0;
}
