class BaseRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(obj_cls):
            cls._registry[name] = obj_cls
            return obj_cls
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(
                f"Unknown {cls.__name__.replace('Registry', '').lower()}: {name}"
            )
        return cls._registry[name](**kwargs)



    @classmethod
    def available(cls):
        return list(cls._registry.keys())
