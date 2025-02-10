from typing import List
from datetime import datetime
from ..abcs.database_types import (
    DbPath,
    Timestamp,
    TimestampTZ,
    Float,
    Decimal,
    Integer,
    TemporalType,
    Text,
    FractionalType,
    Boolean,
    Date,
)
from typing import Dict
from ..abcs.mixins import AbstractMixin_MD5, AbstractMixin_NormalizeValue, AbstractMixin_Schema
from .base import BaseDialect, ThreadedDatabase, import_helper, ConnectError, Mixin_Schema
from ..abcs import Compilable
from ..queries import this, table, Select, SKIP
from ..queries.ast_classes import ForeignKey, TablePath
from .base import TIMESTAMP_PRECISION_POS, Mixin_RandomSample

SESSION_TIME_ZONE = None  # Changed by the tests


@import_helper("mssql")
def import_mssql():
    import pymssql

    return pymssql


class Mixin_MD5(AbstractMixin_MD5):
    def md5_as_int(self, s: str) -> str:
        return f"CONVERT(decimal(38,0), CONVERT(bigint, HashBytes('MD5', {s}), 2))"

class Mixin_NormalizeValue(AbstractMixin_NormalizeValue):
    def normalize_timestamp(self, value: str, coltype: TemporalType) -> str:
        timestamp = f"convert(varchar(26), {value} AT TIME ZONE 'UTC', 25)"
        return (
            f"LEFT({timestamp} + REPLICATE(' ', {coltype.precision}), {TIMESTAMP_PRECISION_POS+6})"
        )

    def normalize_number(self, value: str, coltype: FractionalType) -> str:
        return self.to_string(f"convert(varchar, convert(decimal(38, {coltype.precision}), {value}))")

    def normalize_boolean(self, value: str, _coltype: Boolean) -> str:
        return self.to_string(f"convert(varchar, {value})")
    
class Mixin_Schema(AbstractMixin_Schema):
    def table_information(self) -> TablePath:
        return table("information_schema", "tables")

    def list_tables(self, table_schema: str, like: Compilable = None) -> Select:
        return (
                self.table_information()
                .where(
                    this.table_schema == table_schema if table_schema is not None else SKIP,
                    this.table_name.like(like) if like is not None else SKIP,
                    this.table_type == "BASE TABLE",
                )
                .select(this.table_name)
            )


class MsSQLDialect(BaseDialect, Mixin_Schema):
    name = "MsSQL"
    ROUNDS_ON_PREC_LOSS = True
    SUPPORTS_PRIMARY_KEY = True
    SUPPORTS_INDEXES = True
    MIXINS = {Mixin_Schema, Mixin_MD5, Mixin_NormalizeValue, Mixin_RandomSample}
    AT_TIMEZONE = False

    TYPE_CLASSES = {
        # Numbers
        "tinyint": Integer,
        "smallint": Integer,
        "int": Integer,
        "bigint": Integer,
        "decimal": Decimal,
        "numeric": Decimal,
        "money": Decimal,
        "smallmoney": Decimal,
        "float": Float,
        "real": Float,
        # Timestamps
        "date": Date,
        "time": Timestamp,
        "datetime2": Timestamp,
        "datetimeoffset": TimestampTZ,
        "datetime": Timestamp,
        "smalldatetime": Date,
        # Text
        "char": Text,
        "varchar": Text,
        "text": Text,
        "nchar": Text,
        "nvarchar": Text,
        "ntext": Text,
        # Boolean
        "BIT": Boolean,
    }

    # TSQL has EXPLAIN for Azure SQL Data warehouse
    # But not yet included for the regular RDBMS SQL Server
    def explain_as_text(self, query: str) -> str:
        return f"""SET SHOWPLAN_ALL ON;
                GO
                {query}
                GO
                SET SHOWPLAN_ALL ON;
                GO"""

    def quote(self, s: str):
        return f'"{s}"'

    def to_string(self, s: str):
        return f"CONVERT(VARCHAR(MAX), {s})"

    def concat(self, items: List[str]) -> str:
        joined_exprs = ", ".join(items)
        return f"CONCAT({joined_exprs})"

    def _convert_db_precision_to_digits(self, p: int) -> int:
        return super()._convert_db_precision_to_digits(p) - 2

    # Datetime is stored as UTC by default in MsSQL
    # There is no current way to enforce a timezone for a session
    def set_timezone_to_utc(self) -> str:
        return ""

    def current_timestamp(self) -> str:
        return "SYSUTCDATETIME()"

    def type_repr(self, t) -> str:
        if isinstance(t, TimestampTZ):
            return f"datetimeoffset"
        elif isinstance(t, ForeignKey):
            return self.type_repr(t.type)
        elif isinstance(t, type):
            try:
                return {
                    str: "NVARCHAR(MAX)",
                    bool: "BIT",
                    datetime: "datetime2",
                }[t]
            except KeyError:
                return super().type_repr(t)

        super().type_repr(t)
    
class MsSQL(ThreadedDatabase):
    "AKA sql-server"
    dialect = MsSQLDialect()
    SUPPORTS_ALPHANUMS = False
    SUPPORTS_UNIQUE_CONSTAINT = True
    CONNECT_URI_HELP = "pymssql://<user>:<password>@<host>:<port>/<database>"
    CONNECT_URI_PARAMS = ["database"]

    def __init__(self, host, port, user, password, *, database, thread_count, **kw):
        args = dict(server=host, port=port, database=database, user=user, password=password, conn_properties=['SET QUOTED_IDENTIFIER ON;'], **kw)
        self._args = {k: v for k, v in args.items() if v is not None}

        super().__init__(thread_count=thread_count)

    def create_connection(self):
        self.mssql = import_mssql()
        try:
            return self.mssql.connect(**self._args)
        except self.mssql.Error as e:
            raise ConnectError(*e.args) from e
    
    def select_table_schema(self, path: DbPath) -> str:
        """Provide SQL for selecting the table schema as (name, type, date_prec, num_prec)"""
        
        schema, name = self._normalize_table_path(path)
        if schema == None:
            sql_code = (
                "SELECT column_name, data_type, datetime_precision, numeric_precision, numeric_scale "
                "FROM information_schema.COLUMNS "
                f"WHERE table_name = '{name}'"
            )
        else:
            sql_code = (
                "SELECT column_name, data_type, datetime_precision, numeric_precision, numeric_scale "
                "FROM information_schema.COLUMNS "
                f"WHERE table_name = '{name}' AND table_schema = '{schema}'"
            )

        return sql_code

    def query_table_schema(self, path: DbPath) -> Dict[str, tuple]:
        rows = self.query(self.select_table_schema(path), list)
        if not rows:
            raise RuntimeError(f"{self.name}: Table '{'.'.join(path)}' does not exist, or has no columns")

        d = {r[0]: r for r in rows}
        assert len(d) == len(rows)
        return d