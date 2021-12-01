class AttrProxy:
    def __init__(self, attr_name: str):
        self._attr_name = attr_name

    def _init_done(self):
        self._is_init_done = True

    def __getattr__(self, item):
        return object.__geattr__(object.__geattr__(self, self._attr_name), item)

    def __setattr__(self, key, value):
        if "_is_init_done" in self.__dict__ and key not in self.__dict__:
            setattr(getattr(self, self._attr_name), key, value)
        else:
            object.__setattr__(self, key, value)
