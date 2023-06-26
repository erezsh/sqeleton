from typing import Generator

from ..abcs import DbPath, DbKey
from ..schema import Schema


class T_SKIP:
    def __repr__(self):
        return "SKIP"

    def __deepcopy__(self, memo):
        return self


SKIP = T_SKIP()


class SqeletonError(Exception):
    pass


def args_as_tuple(exprs):
    if len(exprs) == 1:
        (e,) = exprs
        if isinstance(e, Generator):
            return tuple(e)
    return exprs
