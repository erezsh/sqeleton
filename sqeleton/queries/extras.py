"Useful AST classes that don't quite fall within the scope of regular SQL"

from typing import Callable, Sequence
from runtype import dataclass

from ..abcs.database_types import ColType, Native_UUID

from .compiler import Compiler, md
from .ast_classes import Expr, ExprNode, Concat, Code




@dataclass
class NormalizeAsString(ExprNode):
    expr: ExprNode
    expr_type: ColType = None
    type = str


@dataclass
class ApplyFuncAndNormalizeAsString(ExprNode):
    expr: ExprNode
    apply_func: Callable = None


@dataclass
class Checksum(ExprNode):
    exprs: Sequence[Expr]


class Compiler(Compiler):
    @md
    def compile_node(c: Compiler, n: NormalizeAsString) -> str:
        expr = c.compile(n.expr)
        return c.dialect.normalize_value_by_type(expr, n.expr_type or n.expr.type)


    @md
    def compile_node(c: Compiler, n: ApplyFuncAndNormalizeAsString) -> str:
        expr = n.expr
        expr_type = expr.type

        if isinstance(expr_type, Native_UUID):
            # Normalize first, apply template after (for uuids)
            # Needed because min/max(uuid) fails in postgresql
            expr = NormalizeAsString(expr, expr_type)
            if n.apply_func is not None:
                expr = n.apply_func(expr)  # Apply template using Python's string formatting

        else:
            # Apply template before normalizing (for ints)
            if n.apply_func is not None:
                expr = n.apply_func(expr)  # Apply template using Python's string formatting
            expr = NormalizeAsString(expr, expr_type)

        return c.compile(expr)


    @md
    def compile_node(c: Compiler, n: Checksum) -> str:
        if len(n.exprs) > 1:
            exprs = [Code(f"coalesce({c.compile(expr)}, '<null>')") for expr in n.exprs]
            # exprs = [c.compile(e) for e in exprs]
            expr = Concat(exprs, "|")
        else:
            # No need to coalesce - safe to assume that key cannot be null
            (expr,) = n.exprs
        expr = c.compile(expr)
        md5 = c.dialect.md5_as_int(expr)
        return f"sum({md5})"
