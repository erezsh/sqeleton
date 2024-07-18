"""Expressions bound to a specific database"""

import inspect
from functools import wraps
from typing import Union, TYPE_CHECKING

from runtype import dataclass

from .abcs import AbstractDatabase, AbstractCompiler
from .queries.ast_classes import ExprNode, TablePath, Compilable
from .queries.api import table
from .schema import create_schema


@dataclass
class BoundNode(ExprNode):
    database: AbstractDatabase
    node: Compilable

    def __getattr__(self, attr):
        value = getattr(self.node, attr)
        if inspect.ismethod(value):

            @wraps(value)
            def bound_method(*args, **kw):
                return BoundNode(self.database, value(*args, **kw))

            return bound_method
        return value

    def query(self, res_type=list):
        return self.database.query(self.node, res_type=res_type)

    @property
    def type(self):
        return self.node.type

    def compile(self, c: AbstractCompiler) -> str:
        assert c.database is self.database
        return self.node.compile(c)


def bind_node(node, database):
    return BoundNode(database, node)


ExprNode.bind = bind_node


@dataclass
class BoundTable(BoundNode):  # ITable
    database: AbstractDatabase
    node: TablePath

    def with_schema(self, schema):
        table_path = self.node.replace(schema=schema)
        return self.replace(node=table_path)

    def query_schema(self, *, refine: bool = True, refine_where = None, case_sensitive=True):
        table_path = self.node

        if table_path.schema:
            return self

        raw_schema = self.database.query_table_schema(table_path.path)
        schema, _samples = self.database.process_query_table_schema(table_path.path, raw_schema, refine, refine_where)
        schema = create_schema(self.database, table_path.path, schema, case_sensitive)
        return self.with_schema(schema)

    @property
    def schema(self):
        return self.node.schema


def bound_table(database: AbstractDatabase, table_path: Union[TablePath, str, tuple], **kw):
    return BoundTable(database, table(table_path, **kw))


if TYPE_CHECKING:
    class BoundTable(BoundTable, TablePath):
        pass
