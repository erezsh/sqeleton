import contextvars
import decimal
import json
import random
import re
import typing as t
from dataclasses import is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID

from runtype import Dispatch, dataclass

from ..abcs import AbstractCompiler, AbstractDatabase, AbstractDialect, Compilable, DbPath
from ..abcs.mixins import AbstractMixin_Regex, AbstractMixin_TimeTravel
from ..schema import _Field
from ..utils import ArithString, join_iter
from . import ast_classes as ast
from .base import SKIP

cv_params = contextvars.ContextVar("params")


class CompileError(Exception):
    pass


md = Dispatch()


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

        if self._is_root and isinstance(elem, Compilable) and not isinstance(elem, ast.Root):
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
            return self.dialect.uuid_value(elem)
        elif isinstance(elem, ArithString):
            return f"'{elem}'"
        elif isinstance(elem, str):
            return self._add_as_param(elem)
        elif isinstance(elem, (bytes, bytearray)):
            return self._add_as_param(elem.decode())
        elif isinstance(elem, Compilable):
            # return elem.compile(self.replace(_is_root=False))
            return self.replace(_is_root=False).compile_node(elem)
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

    @md
    def compile_node(self, c: ast.Code) -> str:
        if not c.args:
            return c.code

        args = {k: self.compile(v) for k, v in c.args.items()}
        return c.code.format(**args)

    @md
    def compile_node(self, c: ast.Alias) -> str:
        return f"{self.compile(c.expr)} AS {self.quote(c.name)}"

    @md
    def compile_node(self, c: ast.Concat) -> str:
        # We coalesce because on some DBs (e.g. MySQL) concat('a', NULL) is NULL
        # TODO expression can be simpler?
        items = [
            f"coalesce({self.compile(ast.Code(self.dialect.to_string(self.compile(expr))))}, '<null>')"
            for expr in c.exprs
        ]
        assert items
        if len(items) == 1:
            return items[0]

        if c.sep:
            items = list(join_iter(f"'{c.sep}'", items))
        return self.dialect.concat(items)

    @md
    def compile_node(self, c: ast.Count) -> str:
        expr = self.compile(c.expr) if c.expr else "*"
        if c.distinct:
            return f"count(distinct {expr})"

        return f"count({expr})"

    @md
    def compile_node(self, n: ast.TestRegex) -> str:
        if not isinstance(self.dialect, AbstractMixin_Regex):
            raise NotImplementedError(f"No regex implementation for database '{self.database}'")
        regex = self.dialect.test_regex(n.string, n.pattern)
        return self.compile(regex)

    @md
    def compile_node(self, c: ast.Func) -> str:
        args = ", ".join(self.compile(e) for e in c.args)
        return f"{c.name}({args})"

    @md
    def compile_node(self, n: ast.WhenThen) -> str:
        return f"WHEN {self.compile(n.when)} THEN {self.compile(n.then)}"

    @md
    def compile_node(self, c: ast.CaseWhen) -> str:
        when_thens = " ".join(self.compile(case) for case in c.cases)
        else_expr = (" ELSE " + self.compile(c.else_expr)) if c.else_expr is not None else ""
        return f"CASE {when_thens}{else_expr} END"

    @md
    def compile_node(self, n: ast.IsDistinctFrom) -> str:
        return self.dialect.is_distinct_from(self.compile(n.a), self.compile(n.b))

    @md
    def compile_node(self, n: ast.BinOp) -> str:
        expr = f" {n.op} ".join(self.compile(a) for a in n.args)
        return f"({expr})"

    @md
    def compile_node(self, n: ast.UnaryOp) -> str:
        return f"({n.op}{self.compile(n.expr)})"

    @md
    def compile_node(self, n: ast.Column) -> str:
        if self._table_context:
            if len(self._table_context) > 1:
                possible_owners = [t for t in self._table_context if t.schema is None or n.name in t.schema]
                if len(possible_owners) > 1:
                    owners = [t for t in possible_owners if t is n.source_table]
                    if owners:
                        (owner,) = owners
                        if isinstance(owner, ast.TablePath):
                            return f"{self.compile(owner)}.{self.quote(n.name)}"
                        elif isinstance(owner, ast.TableAlias):
                            return f"{self.quote(owner.name)}.{self.quote(n.name)}"

                aliases = [
                    t
                    for t in self._table_context
                    if isinstance(t, ast.TableAlias) and (t.source_table is n.source_table or t is n.source_table)
                ]
                if not aliases:
                    return self.quote(n.name)
                elif len(aliases) > 1:
                    names = [a.name for a in aliases]
                    raise CompileError(f"Too many aliases for column {n.name} between tables: {names}")
                (alias,) = aliases

                return f"{self.quote(alias.name)}.{self.quote(n.name)}"

        return self.quote(n.name)

    @md
    def compile_node(self, n: ast.TableAlias) -> str:
        return f"{self.compile(n.source_table)} {self.quote(n.name)}"

    @md
    def compile_node(self, n: ast.Exists) -> str:
        # TODO use context to avoid replace
        self = self.replace(in_select=False)
        return f"EXISTS ({self.compile(n.expr)})"

    @md
    def compile_node(self, n: ast.TablePath) -> str:
        # TODO normalize path?
        return n.to_string(self.dialect)

    @md
    def compile_node(self, n: ast.TableOp) -> str:
        # TODO contextvar for in_select
        c = self.replace(in_select=False)
        table_expr = f"{c.compile(n.table1)} {n.op} {c.compile(n.table2)}"
        if self.in_select:
            table_expr = f"({table_expr})"
        return table_expr

    @md
    def compile_node(self, n: ast.Desc) -> str:
        e = self.compile(n.expr)
        return f"{e} DESC"

    @md
    def compile_node(self, n: ast.Cte) -> str:
        # TODO contextvar for _table_context
        c = self.replace(_table_context=[], in_select=False)
        compiled = c.compile(n.source_table)

        name = n.name or self.new_unique_name()
        name_params = f"{name}({', '.join(n.params)})" if n.params else name
        self._subqueries[name_params] = compiled

        return name

    @md
    def compile_node(self, n: ast._ResolveColumn) -> str:
        return self.compile_node(n._get_resolved())

    @md
    def compile_node(self, n: ast.In):
        elems = ", ".join(map(self.compile, n.list))
        return f"({self.compile(n.expr)} IN ({elems}))"

    @md
    def compile_node(self, n: ast.InTable):
        table = self.replace(in_select=False).compile_node(n.source_table)
        return f"({self.compile(n.expr)} IN ({table}))"

    @md
    def compile_node(self, n: ast.Cast) -> str:
        return f"cast({self.compile(n.expr)} as {self.compile(n.target_type)})"

    @md
    def compile_node(self, n: ast.Random) -> str:
        return self.dialect.random()

    @md
    def compile_node(self, n: ast.Explain):
        return self.dialect.explain_as_text(self.compile(n.select))

    @md
    def compile_node(self, n: ast.CurrentTimestamp):
        return self.dialect.current_timestamp()

    @md
    def compile_node(self, n: ast.TimeTravel):
        assert isinstance(self.dialect, AbstractMixin_TimeTravel)
        return self.compile(
            self.dialect.time_travel(
                n.table, before=n.before, timestamp=n.timestamp, offset=n.offset, statement=n.statement
            )
        )

    @md
    def compile_node(self, n: ast.Param) -> str:
        params = cv_params.get()
        return self._compile(params[n.name])

    @md
    def compile_node(self, n: ast.Commit) -> str:
        return "COMMIT" if not self.database.is_autocommit else SKIP

    @md
    def compile_node(self, n: ast.InsertToTable) -> str:
        if isinstance(n.expr, ast.ConstantTable):
            compiled_rows = [[self.compile(v) for v in r] for r in n.expr.rows]
            expr = self.dialect.immediate_values(compiled_rows)
        else:
            expr = self.compile(n.expr)

        columns = "(%s)" % ", ".join(map(self.quote, n.columns)) if n.columns is not None else ""

        q = f"INSERT INTO {self.compile(n.path)}{columns} {expr}"

        q += self._compile_returning(n)
        return q

    @md
    def compile_node(self, n: ast.UpdateTable) -> str:
        updates = [f"{k} = {self.compile(v)}" for k, v in n.updates.items()]
        update = f"UPDATE {self.compile(n.path)} SET " + ", ".join(updates)

        if n.where_exprs:
            update += " WHERE " + " AND ".join(map(self.compile, n.where_exprs))
        update += self._compile_returning(n)
        return update

    @md
    def compile_node(self, n: ast.DeleteFromTable) -> str:
        delete = f"DELETE FROM {self.compile(n.path)}"
        if n.where_exprs:
            delete += " WHERE " + " AND ".join(map(self.compile, n.where_exprs))

        delete += self._compile_returning(n)
        return delete

    def _compile_returning(self, n: ast.Statement_MaybeReturning):
        if not n.returning_exprs:
            return ""

        columns = ", ".join(map(self.compile, n.returning_exprs))
        return " RETURNING " + columns

    @md
    def compile_node(self, n: ast.TruncateTable) -> str:
        return f"TRUNCATE TABLE {self.compile(n.path)}"

    @md
    def compile_node(self, n: ast.DropTable) -> str:
        ie = "IF EXISTS " if n.if_exists else ""
        return f"DROP TABLE {ie}{self.compile(n.path)}"

    @md
    def compile_node(self, n: ast.CreateTable) -> str:
        ne = "IF NOT EXISTS " if n.if_not_exists else ""
        if n.source_table:
            return f"CREATE TABLE {ne}{self.compile(n.path)} AS {self.compile(n.source_table)}"

        primary_keys = [k for k, v in n.path.schema.items() if isinstance(v, _Field) and v.options.primary_key]

        if n.primary_keys is not None:
            if primary_keys:
                assert n.primary_keys == primary_keys
            else:
                primary_keys = n.primary_keys

        if primary_keys and self.dialect.SUPPORTS_PRIMARY_KEY:
            pks = ", PRIMARY KEY (%s)" % ", ".join(primary_keys)
        else:
            pks = ""

        schema = ", ".join(f"{self.dialect.quote(k)} {self.dialect.type_repr(v)}" for k, v in n.path.schema.items())
        return f"CREATE TABLE {ne}{self.compile(n.path)}({schema}{pks})"

    @md
    def compile_node(self, n: ast.Join) -> str:
        tables = [
            t if isinstance(t, ast.TableAlias) else ast.TableAlias(t, self.new_unique_name()) for t in n.source_tables
        ]
        c = self.replace(in_select=True)
        op = " JOIN " if n.op is None else f" {n.op} JOIN "
        joined = op.join(c.compile(t) for t in tables)
        c = self.add_table_context(*tables, in_select=True)

        if n.on_exprs:
            on = " AND ".join(c.compile(e) for e in n.on_exprs)
            res = f"{joined} ON {on}"
        else:
            res = joined

        columns = "*" if not n.columns else ", ".join(map(c.compile, n.columns))
        select = f"SELECT {columns} FROM {res}"

        if self.in_select:
            select = f"({select})"
        return select

    @md
    def compile_node(self, n: ast.GroupBy) -> str:
        if n.values_ is None:
            raise CompileError(".group_by() must be followed by a call to .agg()")
        keys = [str(i + 1) for i in range(len(n.keys_))]
        columns = (n.keys_ or []) + (n.values_ or [])
        if isinstance(n.table, ast.Select) and not n.table.columns and n.table.group_by_exprs is None:
            return self.compile(
                n.table.replace(
                    columns=columns,
                    group_by_exprs=[ast.Code(k) for k in keys],
                    having_exprs=n.having_exprs,
                )
            )

        keys_str = ", ".join(keys)
        columns_str = ", ".join(self.compile(x) for x in columns)
        having_str = " HAVING " + " AND ".join(map(self.compile, n.having_exprs)) if n.having_exprs is not None else ""
        select = f"SELECT {columns_str} FROM {SelectCompiler(self).compile(n.table)} GROUP BY {keys_str}{having_str}"

        if self.in_select:
            select = f"({select})"
        return select

    @md
    def compile_node(self, n: ast.Select) -> str:
        c = SelectCompiler(self)

        if isinstance(n.table, (ast.TablePath,)):
            c = c.add_table_context(n.table)

        columns = ", ".join(map(c.compile, n.columns)) if n.columns else "*"
        distinct = "DISTINCT " if n.distinct else ""
        optimizer_hints = self.dialect.optimizer_hints(n.optimizer_hints) if n.optimizer_hints else ""
        select = f"SELECT {optimizer_hints}{distinct}{columns}"

        if n.table:
            select += " FROM " + c.compile(n.table)
        elif self.dialect.PLACEHOLDER_TABLE:
            select += f" FROM {self.dialect.PLACEHOLDER_TABLE}"

        if n.where_exprs:
            select += " WHERE " + " AND ".join(map(c.compile, n.where_exprs))

        if n.group_by_exprs:
            select += " GROUP BY " + ", ".join(map(c.compile, n.group_by_exprs))

        if n.having_exprs:
            assert n.group_by_exprs
            select += " HAVING " + " AND ".join(map(c.compile, n.having_exprs))

        if n.order_by_exprs:
            select += " ORDER BY " + ", ".join(map(c.compile, n.order_by_exprs))

        if n.limit_expr is not None:
            select += " " + self.dialect.offset_limit(0, n.limit_expr)

        if n.postfix:
            select += n.postfix

        if self.in_select:
            select = f"({select})"
        return select


@dataclass
class SelectCompiler(AbstractCompiler):
    c: Compiler

    def compile(self, elem: t.Any, params: t.Dict[str, t.Any] = None) -> str:
        if isinstance(elem, (ast.Select, ast.TableOp, ast.GroupBy, ast.Join)):
            elem = ast.TableAlias(elem, self.c.new_unique_name())
        c = self.c.replace(in_select=True)
        return c.compile(elem, params)

    @property
    def dialect(self):
        return self.c.dialect

    def add_table_context(self, *tables: t.Sequence, **kw):
        return SelectCompiler(self.c.add_table_context(*tables, **kw))
