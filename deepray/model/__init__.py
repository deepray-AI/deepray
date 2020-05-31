def list_available() -> list:
    from deepray.model.model_ctr import BaseCTRModel
    from deepray.utils.list_recursive_subclasses import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BaseCTRModel)
