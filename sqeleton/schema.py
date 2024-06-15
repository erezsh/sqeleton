import logging
from typing import Any, Union, Type

from runtype import dataclass

from .utils import CaseAwareMapping, CaseInsensitiveDict, CaseSensitiveDict
from .abcs import AbstractDatabase, DbPath

logger = logging.getLogger("schema")

Schema = CaseAwareMapping

class TableType:
    pass
    # TODO: This should replace the current Schema type

    @classmethod
    def is_superclass(cls, t):
        return isinstance(t, type) and issubclass(t, cls)


SchemaInput = Union[Type[TableType], Schema, dict]

@dataclass
class Options:
    default: Any = None
    primary_key: bool = False
    auto: bool = False
    # TODO: foreign_key, unique
    # TODO: index?

@dataclass
class _Field:
    type: type
    options: Options

class _Schema(CaseAwareMapping[Union[type, _Field]]):
    pass

    @classmethod
    def make(cls, schema: SchemaInput):
        assert schema
        if TableType.is_superclass(schema):
            def _make_field(k: str, v: type):
                field = getattr(schema, k)
                if field:
                    if not isinstance(field, Options):
                        field = Options(default=v)
                    return _Field(v, field)
                return v

            schema = CaseSensitiveDict({k:_make_field(k, v) for k,v in schema.__annotations__.items()})

        elif isinstance(schema, CaseAwareMapping):
            pass
        else:
            assert isinstance(schema, dict), schema
            schema = CaseSensitiveDict(schema)

        return schema

def options(**kw) -> Any:   # Any, so that type-checking doesn't complain
    return Options(**kw)


def create_schema(db: AbstractDatabase, table_path: DbPath, schema: dict, case_sensitive: bool) -> CaseAwareMapping:
    if case_sensitive:
        return CaseSensitiveDict(schema)

    if len({k.lower() for k in schema}) < len(schema):
        logger.warning(f'Ambiguous schema for {db}:{".".join(table_path)} | Columns = {", ".join(list(schema))}')
        # logger.warning("We recommend to disable case-insensitivity (set --case-sensitive).")
    return CaseInsensitiveDict(schema)
