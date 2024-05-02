from abc import abstractmethod


def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def abstractproperty():
    return lambda f: property(abstractmethod(f))