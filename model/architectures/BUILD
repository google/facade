load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(default_visibility = [
    "//security/brainauth/facade2/model:__subpackages__",
])

pytype_strict_library(
    name = "token_embeddings",
    srcs = ["token_embeddings.py"],
    deps = [
        "//file/recordio/python:recordio",
        "//pyglib:gfile",
        "//security/brainauth/facade2/common/vocab:vocab_py_pb2",
        "//security/brainauth/facade2/model/common:write_once_sorted_dict",
        "//security/brainauth/facade2/model/layers:oov_dropout",
        "//security/brainauth/facade2/model/proto:config_py_pb2",
        "//third_party/py/absl/logging",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//util/hash/python:fingerprint2011",
    ],
)

py_strict_test(
    name = "token_embeddings_test",
    srcs = ["token_embeddings_test.py"],
    deps = [
        ":token_embeddings",
        ":token_embeddings_test_utils",
        "//security/brainauth/facade2/model/proto:config_py_pb2",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "token_embeddings_test_utils",
    testonly = 1,
    srcs = ["token_embeddings_test_utils.py"],
    deps = [
        "//file/recordio/python:recordio",
        "//security/brainauth/facade2/common/vocab:vocab_py_pb2",
    ],
)

pytype_strict_library(
    name = "concatenate_then_snn",
    srcs = ["concatenate_then_snn.py"],
    deps = [
        "//net/proto2/python/public",
        "//security/brainauth/facade2/model/layers:dense_importer",
        "//security/brainauth/facade2/model/layers:segment_embedder",
        "//security/brainauth/facade2/model/layers:snn",
        "//security/brainauth/facade2/model/layers:stateless",
        "//security/brainauth/facade2/model/proto:config_py_pb2",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "concatenate_then_snn_test",
    srcs = ["concatenate_then_snn_test.py"],
    deps = [
        ":concatenate_then_snn",
        "//security/brainauth/facade2/model/proto:config_py_pb2",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
