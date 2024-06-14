import random
from datetime import datetime, date
from typing import Any, Dict, Sequence, List, Optional, Tuple
from uuid import UUID
import decimal
import re
import contextvars
from dataclasses import is_dataclass 
import json
from enum import Enum

from runtype import dataclass

from ..utils import ArithString
from ..abcs import AbstractDatabase, AbstractDialect, DbPath, AbstractCompiler, Compilable
from .base import SKIP


cv_params = contextvars.ContextVar("params")


class CompileError(Exception):
    pass


class Root:
    "Nodes inheriting from Root can be used as root statements in SQL (e.g. SELECT yes, RANDOM() no)"


@dataclass
class CompiledCode:
    code: str
    args: List[Any]
    type: Optional[type]


def eval_template(query_template: str, data_dict: Dict[str, Any], arg_symbol) -> Tuple[str, list]:
    args = []

    def replace_match(match):
        varname = match.group(1)
        args.append(data_dict[varname])
        return arg_symbol

    return re.sub("\xff" + r"\[(\w+)\]", replace_match, query_template), args


@dataclass
class Compiler(AbstractCompiler):
    database: AbstractDatabase
    in_select: bool = False  # Compilation runtime flag
    in_join: bool = False  # Compilation runtime flag

    _table_context: List = []  # List[ITable]
    _subqueries: Dict[str, Any] = {}  # XXX not thread-safe
    _args: Dict[str, Any] = {}
    _args_enabled: bool = False
    _is_root: bool = True

    _counter: List = [0]

    @property
    def dialect(self) -> AbstractDialect:
        return self.database.dialect

    def compile(self, elem: Any, params: Optional[Dict[str, Any]] = None) -> str:
        if params:
            cv_params.set(params)

        if self._is_root and isinstance(elem, Compilable) and not isinstance(elem, Root):
            from .ast_classes import Select

            elem = Select(columns=[elem])

        res = self._compile(elem)
        if self._is_root and self._subqueries:
            subq = ", ".join(f"\n  {k} AS ({v})" for k, v in self._subqueries.items())
            self._subqueries.clear()
            return f"WITH {subq}\n{res}"

        return res

    def compile_with_args(self, elem: Any, params: Optional[Dict[str, Any]] = None) -> CompiledCode:
        assert self._is_root

        if self.dialect.ARG_SYMBOL is not None:
            # Only enable if the database supports args. Otherwise compile normally
            self = self.replace(_args_enabled=True)

        res = self.compile(elem, params)
        if res is SKIP:
            return SKIP

        if self._args:
            res, args = eval_template(res, self._args, self.dialect.ARG_SYMBOL)
            self._args.clear()
        else:
            args = []

        return CompiledCode(res, args, elem.type)

    def _add_as_param(self, elem):
        if self._args_enabled:
            name = self.new_unique_name()
            self._args[name] = elem
            return f"\xff[{name}]"

        if isinstance(elem, bytes):
            return f"b'{elem.decode()}'"
        elif isinstance(elem, bytearray):
            return f"'{elem.decode()}'"
        elif isinstance(elem, (str, UUID)):
            escaped = elem.replace("'", "''")
            return f"'{escaped}'"

        raise NotImplementedError()

    def _compile(self, elem) -> str:
        if elem is None:
            return "NULL"
        elif isinstance(elem, UUID):
            return f"'{elem}'"
        elif isinstance(elem, ArithString):
            return f"'{elem}'"
        elif isinstance(elem, str):
            return self._add_as_param(elem)
        elif isinstance(elem, (bytes, bytearray)):
            return self._add_as_param(elem.decode())
        elif isinstance(elem, Compilable):
            return elem.compile(self.replace(_is_root=False))
        elif isinstance(elem, (int, float)):
            return str(elem)
        elif isinstance(elem, datetime):
            return self.dialect.timestamp_value(elem)
        elif isinstance(elem, date):
            return self._compile(str(elem))
        elif isinstance(elem, decimal.Decimal):
            return str(elem)
        elif elem is ...:
            return "*"
        elif is_dataclass(elem):
            return self._add_as_param(json.dumps(elem.json()))
        elif isinstance(elem, (list, dict)):
            return self._add_as_param(json.dumps(elem))
        elif isinstance(elem, Enum):
            return self._add_as_param(elem.value)
        
        assert False, elem

    def new_unique_name(self, prefix="tmp"):
        self._counter[0] += 1
        return f"{prefix}{self._counter[0]}"

    def new_unique_table_name(self, prefix="tmp") -> DbPath:
        self._counter[0] += 1
        return self.database.parse_table_name(f"{prefix}{self._counter[0]}_{'%x'%random.randrange(2**32)}")

    def add_table_context(self, *tables: Sequence, **kw):
        new_context = self._table_context + list(filter(None, tables))
        if len({t.name for t in new_context}) < len(new_context):
            raise ValueError("Duplicate table alias", {t.name for t in new_context})
        return self.replace(_table_context=new_context, **kw)

    def quote(self, s: str):
        return self.dialect.quote(s)
