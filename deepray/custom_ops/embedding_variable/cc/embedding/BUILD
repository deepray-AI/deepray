load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_copts",
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "embedding_ops_lib",
    deps = [
        ":embedding_gpu",
        ":embedding_var",
        ":embedding_var_ckpt_data",
        ":multi_tier_storage",
        ":ssd_record_descriptor",
    ],
)

cc_library(
    name = "ssd_record_descriptor",
    srcs = ["ssd_record_descriptor.cc"],
    hdrs = [
        "counter_filter_descriptor_impl.h",
        "dynamic_dim_feature_descriptor_impl.h",
        "embedding_config.h",
        "embedding_memory_pool.h",
        "embedding_var_dump_iterator.h",
        "feature_descriptor.h",
        "feature_descriptor_impl.h",
        "hbm_multi_tier_feature_descriptor.h",
        "kv_interface.h",
        "normal_feature_descriptor.h",
        "ssd_record_descriptor.h",
    ],
    copts = [
        "-Wno-unused-result",
        "-Wno-c++11-narrowing",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/lib:allocator",
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/embedding_variable:save_restore_tensor_ev",
        # "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:tf_header_lib",
        "@sparsehash_c11//:dense_hash_map",
    ],
)

cuda_library(
    name = "multi_tier_storage",
    srcs = [
        "multi_tier_storage.cu.cc",
    ],
    hdrs = [
        "bloom_filter_policy.h",
        "cache.h",
        "cache_factory.h",
        "cache_thread_pool_creator.h",
        "counter_filter_descriptor_impl.h",
        "counter_filter_policy.h",
        "cpu_hash_map_kv.h",
        "dram_leveldb_storage.h",
        "dram_pmem_storage.h",
        "dram_ssd_storage.h",
        "dynamic_dim_feature_descriptor_impl.h",
        "emb_file.h",
        "emb_file_creator.h",
        "embedding_config.h",
        "embedding_memory_pool.h",
        "embedding_var.h",
        "embedding_var_ckpt_data.h",
        "embedding_var_context.h",
        "embedding_var_dump_iterator.h",
        "embedding_var_restore.h",
        "eviction_manager.h",
        "feature_descriptor.h",
        "feature_descriptor_impl.h",
        "filter_factory.h",
        "filter_policy.h",
        "globalstep_shrink_policy.h",
        "gpu_hash_map_kv.h",
        "hbm_dram_ssd_storage.h",
        "hbm_dram_storage.h",
        "hbm_multi_tier_feature_descriptor.h",
        "hbm_storage_iterator.h",
        "intra_thread_copy_id_allocator.h",
        "kv_interface.h",
        "l2weight_shrink_policy.h",
        "leveldb_kv.h",
        "multi_tier_storage.h",
        "normal_feature_descriptor.h",
        "nullable_filter_policy.h",
        "shrink_policy.h",
        "single_tier_storage.h",
        "ssd_hash_kv.h",
        "ssd_record_descriptor.h",
        "storage.h",
        "storage_config.h",
        "storage_factory.h",
    ],
    copts = [
        "-Wno-unused-result",
    ],
    deps = [
        ":embedding_gpu",
        "//deepray/custom_ops/embedding_variable/cc/lib:allocator",
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/utils:spin_rw_lock",
        "@com_github_google_leveldb//:leveldb",
        # "@org_tensorflow//tensorflow/core:framework_headers_lib",
        # "@org_tensorflow//tensorflow/core/common_runtime:core_cpu",
        # "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_runtime",
        # "@org_tensorflow//tensorflow/core/kernels:gpu_device_array",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:tf_header_lib",
        "@sparsehash_c11//:dense_hash_map",
    ],
)

cc_library(
    name = "embedding_var_ckpt_data",
    srcs = ["embedding_var_ckpt_data.cc"],
    hdrs = [
        "counter_filter_descriptor_impl.h",
        "dynamic_dim_feature_descriptor_impl.h",
        "embedding_config.h",
        "embedding_memory_pool.h",
        "embedding_var_ckpt_data.h",
        "embedding_var_dump_iterator.h",
        "feature_descriptor.h",
        "feature_descriptor_impl.h",
        "hbm_multi_tier_feature_descriptor.h",
        "kv_interface.h",
        "normal_feature_descriptor.h",
    ],
    copts = [
        "-Wno-c++11-narrowing",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/lib:allocator",
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/embedding_variable:save_restore_tensor_ev",
        # "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_lib",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:tf_header_lib",
        "@sparsehash_c11//:dense_hash_map",
    ],
)

cuda_library(
    name = "embedding_gpu",
    srcs = [
        "batch.cu.cc",
        "gpu_hash_table.cu.cc",
    ],
    hdrs = [
        "batch.h",
        "gpu_hash_table.h",
    ],
    copts = tf_copts(allow_exceptions = True) + if_cuda([
        "-DNV_CUDNN_DISABLE_EXCEPTION",
    ]) + select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options=relaxed-constexpr",
            #"-nvcc_options=ftz=true",
        ],
        "@local_config_cuda//cuda:using_clang": [
            "-fcuda-flush-denormals-to-zero",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "@com_github_google_leveldb//:leveldb",
        "@cuCollections//:cuco_hash_table",
        "@libcuckoo",
        # "@org_tensorflow//tensorflow/core:framework_headers_lib",
        # "@org_tensorflow//tensorflow/core/platform:stream_executor",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:tf_header_lib",
        "@sparsehash_c11//:dense_hash_map",
    ],
    alwayslink = 1,
)

cuda_library(
    name = "embedding_var",
    srcs = [
        "embedding_var.cu.cc",
        "embedding_var_restore.cc",
    ],
    hdrs = [
        "bloom_filter_policy.h",
        "cache.h",
        "cache_factory.h",
        "cache_thread_pool_creator.h",
        "counter_filter_descriptor_impl.h",
        "counter_filter_policy.h",
        "cpu_hash_map_kv.h",
        "dram_leveldb_storage.h",
        "dram_pmem_storage.h",
        "dram_ssd_storage.h",
        "dynamic_dim_feature_descriptor_impl.h",
        "emb_file.h",
        "emb_file_creator.h",
        "embedding_config.h",
        "embedding_memory_pool.h",
        "embedding_var.h",
        "embedding_var_context.h",
        "embedding_var_dump_iterator.h",
        "embedding_var_restore.h",
        "eviction_manager.h",
        "feature_descriptor.h",
        "feature_descriptor_impl.h",
        "filter_factory.h",
        "filter_policy.h",
        "globalstep_shrink_policy.h",
        "gpu_hash_map_kv.h",
        "hbm_dram_ssd_storage.h",
        "hbm_dram_storage.h",
        "hbm_multi_tier_feature_descriptor.h",
        "hbm_storage_iterator.h",
        "intra_thread_copy_id_allocator.h",
        "kv_interface.h",
        "l2weight_shrink_policy.h",
        "leveldb_kv.h",
        "normal_feature_descriptor.h",
        "nullable_filter_policy.h",
        "shrink_policy.h",
        "single_tier_storage.h",
        "ssd_hash_kv.h",
        "storage.h",
        "storage_config.h",
        "storage_factory.h",
    ],
    copts = tf_copts() + ["-g"] + select({
        "//conditions:default": [],
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options=relaxed-constexpr",
        ],
        "@local_config_cuda//cuda:using_clang": [
            "-fcuda-flush-denormals-to-zero",
        ],
    }) + [
        "-Wno-unused-result",
    ],
    deps = [
        "//deepray/custom_ops/embedding_variable/cc/lib:allocator",
        ":embedding_gpu",
        ":embedding_var_ckpt_data",
        ":multi_tier_storage",
        ":ssd_record_descriptor",
        "//deepray/custom_ops/embedding_variable:config_proto_cc",
        "//deepray/custom_ops/utils:spin_rw_lock",
        "@com_github_google_leveldb//:leveldb",
        # "@org_tensorflow//tensorflow/core/common_runtime/gpu:gpu_runtime",
        # "@org_tensorflow//tensorflow/core/kernels:gpu_device_array",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:libtensorflow_cc",
        "@local_config_tf//:tf_header_lib",
        "@sparsehash_c11//:dense_hash_map",
    ],
)
