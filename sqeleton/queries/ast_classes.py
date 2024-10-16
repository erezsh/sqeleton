from dataclasses import field
from datetime import datetime
from typing import Any, Generator, List, Optional, Sequence, Union, Dict, Literal, overload
from collections import ChainMap
from functools import lru_cache

from runtype import dataclass as _dataclass, cv_type_checking, isa

from ..utils import ArithString
from ..abcs import Compilable
from ..abcs.database_types import AbstractTable, AbstractDialect
from ..schema import Schema, TableType, Options

from .base import SKIP, DbPath, args_as_tuple, SqeletonError


class Root:
    "Nodes inheriting from Root can be used as root statements in SQL (e.g. SELECT yes, RANDOM() no)"


class QueryBuilderError(SqeletonError):
    pass


class QB_TypeError(QueryBuilderError):
    pass


dataclass = _dataclass(eq=False, order=False)

ellipsis = type(Ellipsis)


def cache(user_function, /):
    'Simple lightweight unbounded cache.  Sometimes called "memoize".'
    # Taken from https://github.com/python/cpython/blob/3.11/Lib/functools.py
    return lru_cache(maxsize=None)(user_function)


class CompilableNode(Compilable):
    "Base class for query expression nodes"

    type: Any = None

    def _dfs_values(self):
        yield self
        for k, vs in self.asdict().items():  # __dict__ provided by runtype.dataclass
            if k == "source_table":
                # Skip data-sources, we're only interested in data-parameters
                continue
            if not isinstance(vs, (list, tuple)):
                vs = [vs]
            for v in vs:
                if isinstance(v, CompilableNode):
                    yield from v._dfs_values()


class ExprNode(CompilableNode):
    "Base class for query expression nodes"

    def cast_to(self, to):
        return Cast(self, to)


# Query expressions can only interact with objects that are an instance of 'Expr'
Expr = Union[ExprNode, str, bytes, bool, int, float, datetime, ArithString, None, dict, list]


@dataclass
class Code(ExprNode, Root):
    """Raw SQL code"""

    code: str
    args: Dict[str, Expr] = None


def _expr_type(e: Expr) -> type:
    if isinstance(e, ExprNode):
        return e.type
    return type(e)


@dataclass
class Alias(ExprNode):
    """An alias of a column"""

    expr: Expr
    name: str

    @property
    def type(self):
        return _expr_type(self.expr)


def _drop_skips(exprs):
    return [e for e in exprs if e is not SKIP]


def _drop_skips_dict(exprs_dict):
    return {k: v for k, v in exprs_dict.items() if v is not SKIP}


class ITable(AbstractTable):
    """Interface for tabular objects, with focus on SQL operations"""

    source_table: Any
    schema: Schema = None

    def select(self, *exprs: Expr, distinct: bool = SKIP, optimizer_hints=SKIP, **named_exprs) -> "Select":
        """Create a new table with the specified fields"""
        exprs = args_as_tuple(exprs)
        exprs = _drop_skips(exprs)
        named_exprs = _drop_skips_dict(named_exprs)
        exprs += _named_exprs_as_aliases(named_exprs)
        resolve_names(self.source_table, exprs)
        return Select.make(self, columns=exprs, distinct=distinct, optimizer_hints=optimizer_hints)

    def where(self, *exprs) -> "Select":
        """Create a filter for the table, using a WHERE clause.

        Parameters:
            exprs: A list of expressions to filter the table with. Uses of ``SKIP`` will be ignored.

        Returns:
            A new table expression with the filter applied.

        """
        exprs = args_as_tuple(exprs)
        exprs = _drop_skips(exprs)
        if not exprs:
            return self

        resolve_names(self.source_table, exprs)
        return Select.make(self, where_exprs=exprs)

    def order_by(self, *exprs):
        """Order the table by the given expressions, using the ORDER BY clause.

        Parameters:
            exprs: A list of expressions to order by. Uses of ``SKIP`` will be ignored.

        Returns:
            A new table expression with the order applied.
        """
        exprs = args_as_tuple(exprs)
        exprs = _drop_skips(exprs)
        if not exprs:
            return self

        resolve_names(self.source_table, exprs)
        return Select.make(self, order_by_exprs=exprs)

    def limit(self, limit: int):
        """Limit the number of rows returned by the query, using the LIMIT clause.

        Parameters:
            limit: The maximum number of rows to return. ``SKIP`` will set no limit.
        """
        if limit is SKIP:
            return self

        return Select.make(self, limit_expr=limit)

    def join(self, target: "ITable") -> "Join":
        """Join this table with the target table.

        Parameters:
            target: The table to join with.

        Returns:
            A new table expression representing the join.
        """
        return Join([self, target])

    def group_by(self, *keys) -> "GroupBy":
        """Group according to the given keys, using the GROUP BY clause.

        Must be followed by a call to :ref:``GroupBy.agg()``

        Parameters:
            keys: A list of expressions to group by. Uses of ``SKIP`` will be ignored.

        Returns:
            A new table expression with the group-by applied.
        """
        keys = args_as_tuple(keys)
        keys = _drop_skips(keys)
        resolve_names(self.source_table, keys)

        return GroupBy(self, keys)

    def _get_column(self, name: str):
        if self.schema:
            name = self.schema.get_key(name)  # Get the actual name. Might be case-insensitive.
        return Column(self, name)

    # def __getattr__(self, column):
    #     return self._get_column(column)

    def __getitem__(self, column):
        if isinstance(column, (list, tuple)):
            return [self[c] for c in column]
        elif column is ...:
            assert self.schema
            return [self[k] for k in self.schema]
        if not isinstance(column, str):
            raise TypeError(column)
        return self._get_column(column)

    def count(self) -> "Select":
        """SELECT COUNT(*) FROM self"""
        return Select(self, [Count()])

    def union(self, other: "ITable") -> "TableOp":
        """SELECT * FROM self UNION other"""
        return TableOp("UNION", self, other)

    def union_all(self, other: "ITable") -> "TableOp":
        """SELECT * FROM self UNION ALL other"""
        return TableOp("UNION ALL", self, other)

    def minus(self, other: "ITable") -> "TableOp":
        """SELECT * FROM self EXCEPT other"""
        # aka
        return TableOp("EXCEPT", self, other)

    def intersect(self, other: "ITable") -> "TableOp":
        """SELECT * FROM self INTERSECT other"""
        return TableOp("INTERSECT", self, other)

    def alias(self, name: str) -> "TableAlias":
        """Create an alias for the table. If there already is an alias, it will be replaced."""
        if isinstance(self, TableAlias):
            return self.replace(name=name)
        return TableAlias(self, name)


@dataclass
class Concat(ExprNode):
    """Concatenate expressions, with an optional seperator."""

    exprs: list
    sep: str = None

    type = str


@dataclass
class Count(ExprNode):
    """Count the number of rows in a table"""

    expr: Expr = None
    distinct: bool = False

    type = int


class LazyOps:
    def __add__(self, other):
        return BinOp("+", [self, other])

    def __sub__(self, other):
        return BinOp("-", [self, other])

    def __rsub__(self, other):
        return BinOp("-", [other, self])

    def __mul__(self, other):
        return BinOp("*", [self, other])

    __radd__ = __add__
    __rmul__ = __mul__

    def __truediv__(self, other):
        return BinOp("/", [self, other])

    def __neg__(self):
        return UnaryOp("-", self)

    def not_(self):
        return UnaryOp("NOT ", self)

    def __gt__(self, other):
        return BinBoolOp(">", [self, other])

    def __ge__(self, other):
        return BinBoolOp(">=", [self, other])

    def __eq__(self, other):
        if cv_type_checking.get():
            return super().__eq__(other)

        if other is None:
            return BinBoolOp("IS", [self, None])

        return BinBoolOp("=", [self, other])

    def __ne__(self, other):
        if cv_type_checking.get():
            return super().__ne__(other)

        if other is None:
            return BinBoolOp("IS NOT", [self, None])

        return BinBoolOp("<>", [self, other])

    def __lt__(self, other):
        return BinBoolOp("<", [self, other])

    def __le__(self, other):
        return BinBoolOp("<=", [self, other])

    def __or__(self, other):
        return BinBoolOp("OR", [self, other])

    def __and__(self, other):
        return BinBoolOp("AND", [self, other])

    def is_distinct_from(self, other: Expr) -> ExprNode:
        """Check if the expression is distinct from the other expression (i.e. not equal, treating NULL as a value)"""
        return IsDistinctFrom(self, other)

    def like(self, other: Expr) -> ExprNode:
        """Check if the expression is like the other expression, using SQL LIKE syntax"""
        return BinBoolOp("LIKE", [self, other])

    def ilike(self, other: Expr) -> ExprNode:
        """Check if the expression is like the other expression, case-insensitive, using SQL ILIKE syntax"""
        return BinBoolOp("ILIKE", [self, other])

    def in_(self, *others: Expr) -> ExprNode:
        """Check if the expression is in the list or table given

        Parameters:
            others: A list of constants, or a single table expression

        Returns:
            A new expression representing the 'IN' operation
        """
        others = args_as_tuple(others)
        assert isinstance(others, tuple), f"Only lists of constants are supported for now, not {others}"
        if len(others) == 0:
            return False  # SQL value
        elif len(others) == 1 and isinstance(others[0], ExprTable):
            return InTable(self, others[0])
        return In(self, others)

    def test_regex(self, other: Expr) -> ExprNode:
        """Check if the expression matches the regex pattern.

        Parameters:
            other: A string expression representing the regex pattern.

        Returns:
            A new expression representing the regex test.
        """
        return TestRegex(self, other)

    def sum(self) -> ExprNode:
        """Sum the values of the expression"""
        return Func("SUM", [self])

    def count(self, distinct: bool = False):
        """Count the number of rows in the expression

        Parameters:
            distinct (bool): If True, only count distinct rows.
        """
        return Count(self, distinct=distinct)

    def max(self):
        """Get the maximum value of the list expression"""
        return Func("MAX", [self])

    def min(self):
        """Get the minimum value of the list expression"""
        return Func("MIN", [self])


@dataclass
class TestRegex(ExprNode, LazyOps):
    """Check if the expression matches the regex pattern"""

    string: Expr
    pattern: Expr


@dataclass
class Func(ExprNode, LazyOps):
    """Call a function with the given arguments"""

    name: str
    args: Sequence[Expr]
    ret_type: type = None


@dataclass
class WhenThen(ExprNode):
    """A 'when/then' clause in a case-when expression"""

    when: Expr
    then: Expr


@dataclass
class CaseWhen(ExprNode, LazyOps):
    """A case-when expression"""

    cases: Sequence[WhenThen]
    else_expr: Expr = None

    @property
    def type(self):
        then_types = {_expr_type(case.then) for case in self.cases}
        if self.else_expr:
            then_types.add(_expr_type(self.else_expr))
        if len(then_types) > 1:
            raise QB_TypeError(f"Non-matching types in when: {then_types}")
        (t,) = then_types
        return t

    def when(self, *whens: Expr) -> "QB_When":
        """Add a new 'when' clause to the case expression

        Must be followed by a call to `.then()`
        """
        whens = args_as_tuple(whens)
        whens = _drop_skips(whens)
        if not whens:
            raise QueryBuilderError("Expected valid whens")

        # XXX reimplementing api.and_()
        if len(whens) == 1:
            return QB_When(self, whens[0])
        return QB_When(self, BinBoolOp("AND", whens))

    def else_(self, then: Expr) -> "CaseWhen":
        """Add an 'else' clause to the case expression.

        Can only be called once per case-when!

        Parameters:
            then: The expression to return if none of the 'when' clauses match.

        Returns:
            A new case-when expression with the 'else' clause added.
        """
        if self.else_expr is not None:
            raise QueryBuilderError(f"Else clause already specified in {self}")

        return self.replace(else_expr=then)


@dataclass
class QB_When:
    "Partial case-when, used for query-building"
    casewhen: CaseWhen
    when: Expr

    def then(self, then: Expr) -> CaseWhen:
        """Add a 'then' clause after a 'when' was added."""
        case = WhenThen(self.when, then)
        return self.casewhen.replace(cases=self.casewhen.cases + [case])


@_dataclass(eq=False, order=False)
class IsDistinctFrom(ExprNode, LazyOps):
    """Check if two expressions are distinct from each other (i.e. not equal, treating NULL as a value)"""

    a: Expr
    b: Expr
    type = bool


@_dataclass(eq=False, order=False)
class BinOp(ExprNode, LazyOps):
    """Binary operation on expressions"""

    op: str
    args: Sequence[Expr]

    @property
    def type(self):
        types = {_expr_type(i) for i in self.args}
        if len(types) > 1:
            # raise TypeError(f"Expected all args to have the same type, got {types}")
            return Union[tuple(types)]
        (t,) = types
        return t


@dataclass
class UnaryOp(ExprNode, LazyOps):
    """Unary operation on an expression"""

    op: str
    expr: Expr

    @property
    def type(self):
        return self.expr.type


class BinBoolOp(BinOp):
    """Binary boolean operation on expressions"""

    type = bool


@_dataclass(eq=False, order=False)
class Column(ExprNode, LazyOps):
    """A column in a table"""

    source_table: ITable  # TODO: TablePath
    name: str

    @property
    def type(self):
        if self.source_table.schema is None:
            raise QueryBuilderError(f"Schema required for table {self.source_table}")
        return self.source_table.schema[self.name]


class ExprTable(ExprNode, ITable):
    pass


@_dataclass
class TablePath(ExprTable):
    """A path to a table in a database"""

    path: DbPath
    schema: Optional[Schema] = field(default=None, repr=False)

    @property
    def source_table(self):
        return self

    @property
    def name(self):
        return self.path[-1]

    def to_string(self, dialect: AbstractDialect):
        return ".".join(map(dialect.quote, self.path))

    def __repr__(self) -> str:
        if self.schema:
            return f"TablePath({self.path!r}, schema=<{len(self.schema)} cols>)"
        return f"TablePath({self.path!r})"

    # Statement shorthands
    def create(self, source_table: ITable = None, *, if_not_exists: bool = False, primary_keys: List[str] = None):
        """Returns a query expression to create a new table.

        Parameters:
            source_table: a table expression to use for initializing the table.
                          If not provided, the table must have a schema specified.
            if_not_exists: Add a 'if not exists' clause or not. (note: not all dbs support it!)
            primary_keys: List of column names which define the primary key
        """

        if source_table is None and not self.schema:
            raise ValueError("Either schema or source table needed to create table")
        if isinstance(source_table, TablePath):
            source_table = source_table.select()
        return CreateTable(self, source_table, if_not_exists=if_not_exists, primary_keys=primary_keys)

    def drop(self, if_exists: bool = False):
        """Returns a query expression to delete the table.

        Parameters:
            if_not_exists: Add a 'if not exists' clause or not. (note: not all dbs support it!)
        """
        return DropTable(self, if_exists=if_exists)

    def truncate(self):
        """Returns a query expression to truncate the table. (remove all rows)"""
        return TruncateTable(self)

    def delete_rows(self, *where_exprs: Union[Expr, Literal[SKIP]]) -> "DeleteFromTable":
        """Returns a query expression to delete rows from the table.

        Parameters:
            where_exprs: A list of expressions to filter the rows to delete. Uses of ``SKIP`` will be ignored.
                         If no where expressions are provided, all rows will be deleted.

        Returns:
            A new query expression to delete rows from the table.
        """
        where_exprs = args_as_tuple(where_exprs)
        where_exprs = _drop_skips(where_exprs)
        if not where_exprs:
            return self.truncate()

        resolve_names(self.source_table, where_exprs)
        return DeleteFromTable(self, where_exprs)

    def update_fields(self, *where_exprs: Expr, **kv) -> "UpdateTable":
        """Returns a query expression to update fields in the table.

        Parameters:
            where_exprs: A list of expressions to filter the rows to update. Uses of ``SKIP`` will be ignored.
                         If no where expressions are provided, all rows will be updated.
            kv: A dictionary of column names and values to update. The values can be expressions.

            Returns:
                A new query expression to update fields in the table.
        """
        where_exprs = args_as_tuple(where_exprs)
        where_exprs = _drop_skips(where_exprs)
        resolve_names(self.source_table, where_exprs)
        resolve_names(self.source_table, kv.values())
        return UpdateTable(self, kv, where_exprs)

    def insert_rows(self, rows: Sequence, *, columns: List[str] = None):
        """Returns a query expression to insert rows to the table, given as Python values.

        Parameters:
            rows: A list of tuples. Must all have the same width.
            columns: Names of columns being populated. If specified, must have the same length as the tuples.
        """
        # TODO support expressions (now, random, etc.)
        rows = list(rows)
        if not rows:
            return SKIP

        if isinstance(rows[0], dict):
            # TODO: Validate all rows are the same?
            keys = list(rows[0].keys())
            if not columns:
                columns = keys
            elif not (set(columns) <= set(rows[0].keys())):
                raise ValueError("Keys in dictionary are not a subset of 'columns'")
            rows = [[row.get(k) for k in columns] for row in rows]

        return InsertToTable(self, ConstantTable(rows), columns=columns)

    def insert_row(self, *values, columns: List[str] = None, **kw):
        """Returns a query expression to insert a single row to the table, given as Python values.

        Parameters:
            columns: Names of columns being populated. If specified, must have the same length as 'values'
        """
        if (not values) == (not kw):
            raise ValueError("Must provide either positional arguments or keyword arguments, but not a mix of both.")
        if values:
            if len(values) == 1 and isinstance(values[0], TableType):
                assert columns is None
                kw = {
                    k: v.default if isinstance(v, Options) else v
                    for k, v in values[0]
                    if not (isinstance(v, Options) and v.auto)
                }
            else:
                return InsertToTable(self, ConstantTable([values]), columns=columns)

        assert kw
        assert not columns
        return InsertToTable(self, ConstantTable([list(kw.values())]), columns=list(kw.keys()))

    def insert_expr(self, expr: Expr):
        """Returns a query expression to insert rows to the table, given as a query expression.

        Parameters:
            expr: query expression to from which to read the rows
        """
        if isinstance(expr, TablePath):
            expr = expr.select()
        return InsertToTable(self, expr)

    def time_travel(
        self, *, before: bool = False, timestamp: datetime = None, offset: int = None, statement: str = None
    ) -> Compilable:
        """Selects historical data from the table

        Parameters:
            before: If false, inclusive of the specified point in time.
                     If True, only return the time before it. (at/before)
            timestamp: A constant timestamp
            offset: the time 'offset' seconds before now
            statement: identifier for statement, e.g. query ID

        Must specify exactly one of `timestamp`, `offset` or `statement`.
        """
        if sum(int(i is not None) for i in (timestamp, offset, statement)) != 1:
            raise ValueError("Must specify exactly one of `timestamp`, `offset` or `statement`.")

        if timestamp is not None:
            assert offset is None and statement is None


@_dataclass
class ForeignKey:
    """A foreign key constraint"""

    table: TablePath
    field: str

    @property
    def type(self):
        return self.table.schema[self.field]


@dataclass
class TableAlias(ExprTable):
    """An alias for a table"""

    source_table: ITable
    name: str

    @property
    def schema(self):
        return self.source_table.schema


@dataclass
class Exists(ExprNode, LazyOps):
    """Check if a subquery returns any rows"""

    expr: ITable

    type = bool


SelectColumns = Sequence[Union[Expr, ellipsis]]


def _expand_ellipsis(schema: dict, columns: SelectColumns):
    for c in columns:
        if c is ...:
            # select all, i.e. *
            yield from schema.items()
        else:
            yield c.name, c.type


def _union_dicts(*ds):
    unioned = {}
    for d in ds:
        unioned.update(d)
    return unioned


@dataclass
class Join(ExprNode, ITable, Root):
    """A join operation between two tables"""

    source_tables: Sequence[ITable]
    op: str = None
    on_exprs: Sequence[Expr] = None
    columns: SelectColumns = None

    @property
    def source_table(self):
        return self

    @property
    @cache
    def schema(self):
        if not self.columns:
            schemas = [t.schema for t in self.source_tables if t.schema]
            assert schemas and all(schemas)
            return type(schemas[0])(ChainMap(*schemas))  # TODO merge dictionaries in compliance with SQL dialect!

        # TODO validate types match between both tables
        schemas = [s.schema for s in self.source_tables]
        d = _union_dicts(*schemas)
        return type(schemas[0])(dict(_expand_ellipsis(d, self.columns)))

    def on(self, *exprs) -> "Join":
        """Add an ON clause, for filtering the result of the cartesian product (i.e. the JOIN)"""
        if len(exprs) == 1:
            (e,) = exprs
            if isinstance(e, Generator):
                exprs = tuple(e)

        exprs = _drop_skips(exprs)
        if not exprs:
            return self

        exprs = [BinBoolOp("=", [t[e] for t in self.source_tables]) if isinstance(e, str) else e for e in exprs]

        return self.replace(on_exprs=(self.on_exprs or []) + exprs)

    def select(self, *exprs: Expr, **named_exprs) -> "Join":
        """Select fields to return from the JOIN operation

        See Also: ``ITable.select()``
        """
        for e in exprs:
            if not isa(e, Expr):
                raise TypeError(e)

        if self.columns is not None:
            # join-select already applied
            return ITable.select(self, *exprs, **named_exprs)

        exprs = _drop_skips(exprs)
        named_exprs = _drop_skips_dict(named_exprs)
        exprs += _named_exprs_as_aliases(named_exprs)
        resolve_names(self.source_table, exprs)
        # TODO Ensure exprs <= self.columns ?
        return self.replace(columns=exprs)


@dataclass
class GroupBy(ExprNode, ITable, Root):
    """A group-by operation using the GROUP BY clause"""

    table: ITable
    keys_: Sequence[Expr] = None  # IKey?
    values_: Sequence[Expr] = None
    having_exprs: Sequence[Expr] = None

    @property
    def source_table(self):
        return self

    @property
    @cache
    def schema(self):
        s = self.table.schema
        if s is None:
            return None
        return type(s)({c.name: c.type for c in self.keys_ + self.values_})

    def __post_init__(self):
        assert self.keys_ or self.values_

    def having(self, *exprs) -> "GroupBy":
        """Add a 'HAVING' clause to the group-by

        Parameters:
            exprs: A list of expressions to filter the groups. Uses of ``SKIP`` will be ignored.

        Returns:
            A new group-by expression with the filter applied.
        """
        exprs = args_as_tuple(exprs)
        exprs = _drop_skips(exprs)
        if not exprs:
            return self

        resolve_names(self.table, exprs)
        return self.replace(having_exprs=(self.having_exprs or []) + exprs)

    def agg(self, *exprs, **named_exprs) -> "GroupBy":
        """Select aggregated fields for the group-by.

        Parameters:
            exprs: A list of expressions to aggregate. Uses of ``SKIP`` will be ignored.
            named_exprs: A dictionary of named expressions to aggregate. Uses of ``SKIP`` will be ignored.

        Returns:
            A new group-by expression with the aggregated fields applied.
        """
        exprs = args_as_tuple(exprs)
        exprs = _drop_skips(exprs)

        named_exprs = _drop_skips_dict(named_exprs)
        exprs += _named_exprs_as_aliases(named_exprs)

        resolve_names(self.table, exprs)
        return self.replace(values_=(self.values_ or []) + exprs)


@dataclass
class TableOp(ExprNode, ITable, Root):
    """A binary operation between two tables"""

    op: str
    table1: ITable
    table2: ITable

    @property
    def source_table(self):
        return self

    @property
    def type(self):
        # TODO ensure types of both tables are compatible
        return self.table1.type

    @property
    def schema(self):
        s1 = self.table1.schema
        s2 = self.table2.schema
        assert len(s1) == len(s2)
        return s1


@dataclass
class Desc(ExprNode):
    """A descending order"""

    expr: ExprNode


@dataclass
class Select(ExprTable, Root):
    """A SELECT statement"""

    table: Expr = None
    columns: SelectColumns = None
    where_exprs: Sequence[Expr] = None
    order_by_exprs: Sequence[Expr] = None
    group_by_exprs: Sequence[Expr] = None
    having_exprs: Sequence[Expr] = None
    limit_expr: int = None
    distinct: bool = False
    optimizer_hints: Sequence[Expr] = None
    postfix: str = None

    @property
    @cache
    def schema(self):
        s = self.table.schema
        if s is None or not self.columns:
            return s
        return type(s)(dict(_expand_ellipsis(s, self.columns)))

    @property
    def source_table(self):
        return self

    @classmethod
    def make(cls, table: ITable, distinct: bool = SKIP, optimizer_hints: str = SKIP, **kwargs):
        assert "table" not in kwargs

        if optimizer_hints is not SKIP:
            kwargs["optimizer_hints"] = optimizer_hints
        if distinct is not SKIP:
            kwargs["distinct"] = distinct

        # If table is not a select, return a new Select instance
        if not isinstance(table, cls):
            return cls(table, **kwargs)

        if "columns" in kwargs and table.columns is not None:
            return cls(table, **kwargs)

        # We can safely assume isinstance(table, Select)
        # We will try to merge them, instead of creating nested instances
        if distinct is False and table.distinct:
            # Cannot merge the selects
            return cls(table, **kwargs)

        if table.limit_expr or table.group_by_exprs:
            # Cannot merge the selects
            return cls(table, **kwargs)

        # Fill in missing attributes
        for k, v in kwargs.items():
            if getattr(table, k) is not None:
                if k == "where_exprs":  # Additive attribute
                    kwargs[k] = getattr(table, k) + v
                elif k in ["distinct", "optimizer_hints"]:
                    pass
                else:
                    raise ValueError(k)

        return table.replace(**kwargs)


@dataclass
class Cte(ExprNode, ITable):
    """A common table expression (i.e. shared subquery)"""

    source_table: Expr
    name: str = None
    params: Sequence[str] = None

    @property
    def schema(self):
        # TODO add cte to schema
        return self.source_table.schema


def _named_exprs_as_aliases(named_exprs):
    return [Alias(expr, name) for name, expr in named_exprs.items()]


def resolve_names(source_table, exprs):
    i = 0
    for expr in exprs:
        # Iterate recursively and update _ResolveColumn instances with the right expression
        if isinstance(expr, ExprNode):
            for v in expr._dfs_values():
                if isinstance(v, _ResolveColumn):
                    v.resolve(source_table._get_column(v.resolve_name))
                    i += 1


@_dataclass(frozen=False, eq=False, order=False)
class _ResolveColumn(ExprNode, LazyOps):
    resolve_name: str
    resolved: Expr = None

    def resolve(self, expr: Expr):
        if self.resolved is not None:
            raise QueryBuilderError(f"Column '{self.resolve_name}' Already resolved! To value: {self.resolved}")
        self.resolved = expr

    def _get_resolved(self) -> Expr:
        if self.resolved is None:
            breakpoint()
            raise QueryBuilderError(f"Column not resolved: {self.resolve_name}")
        return self.resolved

    @property
    def type(self):
        return self._get_resolved().type

    @property
    def name(self):
        return self._get_resolved().name

    def __rsub__(self, other):
        if other is ...:
            # return Wildcard()
            pass

        # return super().__rsub__(other)    XXX why does it fail?
        return LazyOps.__rsub__(self, other)


@dataclass
class Wildcard:
    """Wildcard for selecting all columns"""

    exclude: List[str]


class This:
    """Builder object for accessing table attributes.

    Automatically evaluates to the the 'top-most' table during compilation.
    """

    def __getattr__(self, name):
        return _ResolveColumn(name)

    @overload
    def __getitem__(self, name: str) -> "This": ...
    @overload
    def __getitem__(self, name: (list, tuple)) -> List["This"]: ...

    def __getitem__(self, name):
        if isinstance(name, (list, tuple)):
            return [_ResolveColumn(n) for n in name]
        return _ResolveColumn(name)


@dataclass
class In(ExprNode):
    """Check if an expression is in a list of values"""

    expr: Expr
    list: Sequence[Expr]

    type = bool


@dataclass
class InTable(ExprNode):
    """Check if an expression is in a table"""

    expr: Expr
    source_table: ExprTable

    type = bool


@dataclass
class Cast(ExprNode):
    """Cast an expression to a different type"""

    expr: Expr
    target_type: Expr


@dataclass
class Random(ExprNode, LazyOps):
    """A random number"""

    type = float


@dataclass
class ConstantTable(ExprNode):
    """A table of constant values"""

    rows: Sequence[Sequence]


@dataclass
class Explain(ExprNode, Root):
    """An EXPLAIN statement"""

    select: Select

    type = str


@dataclass
class CurrentTimestamp(ExprNode, LazyOps):
    """The current timestamp"""

    type = datetime


@dataclass
class TimeTravel(ITable):
    """A time-travel operation"""

    table: TablePath
    before: bool = False
    timestamp: datetime = None
    offset: int = None
    statement: str = None


# DDL


class Statement(CompilableNode, Root):
    type = None


@dataclass
class CreateTable(Statement):
    """a CREATE TABLE statement"""

    path: TablePath
    source_table: Expr = None
    if_not_exists: bool = False
    primary_keys: List[str] = None


@dataclass
class DropTable(Statement):
    """a DROP TABLE statement"""

    path: TablePath
    if_exists: bool = False


@dataclass
class TruncateTable(Statement):
    """a TRUNCATE TABLE statement"""

    path: TablePath


class Statement_MaybeReturning(Statement):
    returning_exprs: SelectColumns = None

    @property
    def type(self):
        if self.returning_exprs:
            return ITable
        return None

    def returning(self, *exprs):
        """Add a 'RETURNING' clause to the current node.

        Note: Not all databases support this feature!
        """
        if self.returning_exprs:
            raise ValueError("A returning clause has already been specified")

        exprs = args_as_tuple(exprs)
        exprs = _drop_skips(exprs)
        if not exprs:
            return self

        resolve_names(self.path, exprs)
        return self.replace(returning_exprs=exprs)


@dataclass
class DeleteFromTable(Statement_MaybeReturning):
    """a DELETE FROM statement"""

    path: TablePath
    where_exprs: Sequence[Expr] = None
    returning_exprs: SelectColumns = None


@dataclass
class UpdateTable(Statement_MaybeReturning):
    """an UPDATE statement"""

    path: TablePath
    updates: Dict[str, Expr]
    where_exprs: Sequence[Expr] = None
    returning_exprs: SelectColumns = None


@dataclass
class InsertToTable(Statement_MaybeReturning):
    """an INSERT INTO statement"""

    path: TablePath
    expr: Expr
    columns: List[str] = None
    returning_exprs: SelectColumns = None


@dataclass
class Commit(Statement):
    """Generate a COMMIT statement, if we're in the middle of a transaction, or in auto-commit. Otherwise SKIP."""


@dataclass
class Param(ExprNode, ITable):
    """A value placeholder, to be specified at compilation time using the `cv_params` context variable."""

    name: str

    @property
    def source_table(self):
        return self
