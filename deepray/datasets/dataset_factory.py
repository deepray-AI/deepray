from absl import logging, flags

flags.DEFINE_string("data_source", "parquet_dataset", "parquet or tfrecord")

FLAGS = flags.FLAGS

"""
Build model
"""


def load_dataset():
  # Get the module class from the module file name
  module_class_name = FLAGS.data_source
  logging.info(module_class_name)
  if module_class_name == "arrows3_dataset":

    logging.info("Load dataset v3")

    module_instance = ArsenalDatasetV3()


  elif module_class_name == "parquet_dataset":
    from deepray.datasets.parquet_pipeline.ali_parquet_dataset import ParquetPipeLine

    logging.info("Load parquet dataset")
    module_instance = ParquetPipeLine()

  """
    abs_mod_dir_path = os.path.dirname(os.path.realpath(__file__))
    logging.info(f"abs_mod_dir_path: {abs_mod_dir_path}")
    sys.path.insert(0, abs_mod_dir_path)

    # import pkgutil
    # test = [name for _, name, _ in pkgutil.iter_modules([abs_mod_dir_path])]

    # Get's the module's class to call functions on
    module = importlib.import_module(module_class_name)

    # Find the main class in our imported module
    logging.info(list(module.__dict__.items())[-5][0])
    module_main_cls = list(module.__dict__.items())[-5][1]

    # Create's an instance of that module's class
    module_instance = module_main_cls()

    # Invalidate Python's caches so the new modules can be found
    importlib.invalidate_caches()
    """
  return module_instance
