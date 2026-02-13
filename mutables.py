# it's a mutable integer and that's all
class Minteger(object):
    # initialise by setting the value
    def __init__(self, value: int):
        assert isinstance(value, int), f'Minteger is for integers!'
        self._integer: dict[int] = {"value": int(value)}

    @property
    def value(self):
        return self._integer["value"]

    def set(self, new_val: int):
        self.value = int(new_val)

    def inc(self):
        self.value += 1

    def dec(self):
        self.value -= 1

    def __add__(self, b) -> int:
        if isinstance(b, Minteger):
            b = b.value
        return self.value + int(b)

    def __mul__(self, b) -> int:
        if isinstance(b, Minteger):
            b = b.value
        return self.value * b

    def __truediv__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value / b

    def __and__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value & b

    def __neg__(self):
        return - self.value

    def __eq__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value == b

    def __lt__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value < b

    def __le__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value <= b

    def __gt__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value > b

    def __ge__(self, b):
        if isinstance(b, Minteger):
            b = b.value
        return self.value >= b
    