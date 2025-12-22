class TrainerRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(tun_cls):
            cls._registry[name] = tun_cls
            return tun_cls
        return decorator

    @classmethod
    def create(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Unknown tuner: {name}")
        return cls._registry[name](**kwargs)