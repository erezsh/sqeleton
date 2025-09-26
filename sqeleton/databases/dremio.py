from typing import Optional, Union

from sqeleton.queries import this, SKIP
from .base import BaseDialect, QueryResult, import_helper, parse_table_name, ThreadLocalInterpreter, \
    Mixin_RandomSample, logger, ThreadedDatabase, Mixin_Schema, TIMESTAMP_PRECISION_POS
from ..abcs import Compilable
from ..abcs.database_types import (
    Timestamp,
    Integer,
    Float,
    Text,
    FractionalType,
    Date,
    DbPath,
    DbTime,
    Decimal,
    ColType,
    TemporalType,
    Boolean,
    ColType_UUID
)
from ..abcs.mixins import AbstractMixin_NormalizeValue, AbstractMixin_MD5
from ..queries.compiler import CompiledCode


def query_cursor(c, sql_code: CompiledCode) -> Optional[QueryResult]:
    logger.debug(f"[Dremio] Executing SQL: {sql_code.code} || {sql_code.args}")
    c.execute(sql_code.code, sql_code.args)

    columns = c.description and [col[0] for col in c.description]
    return QueryResult(c.fetchall(), columns)


@import_helper("dremio")
def import_dremio():
    import sqlalchemy

    return sqlalchemy


class Mixin_MD5(AbstractMixin_MD5):
    def md5_as_int(self, s: str) -> str:
        return f"CAST(CONV(SUBSTR(MD5({s}), 18), 16, 10) AS BIGINT)"


class Mixin_NormalizeValue(AbstractMixin_NormalizeValue):

    def normalize_uuid(self, value: str, coltype: ColType_UUID) -> str:
        # Trim doesn't work on CHAR type
        return f"TRIM(CAST({value} AS VARCHAR))"

    def normalize_timestamp(self, value: str, coltype: TemporalType) -> str:
        """Dremio timestamp contains no more than 6 digits in precision"""
        precision_format = ('F' * coltype.precision) if coltype.precision > 0 else ''
        return f"rpad(TO_CHAR(CAST({value} AS TIMESTAMP), 'YYYY-MM-DD HH24:MI:SS.{precision_format}'), {TIMESTAMP_PRECISION_POS + 6}, '0')"

    def normalize_number(self, value: str, coltype: FractionalType) -> str:
        if coltype.precision > 0:
            value = f"format_number({value}, {coltype.precision})"
        return f"replace({self.to_string(value)}, ',', '')"

    def normalize_boolean(self, value: str, _coltype: Boolean) -> str:
        return self.to_string(f"cast({value} as int)")

class Dialect(BaseDialect, Mixin_Schema):
    name = "Dremio"
    ROUNDS_ON_PREC_LOSS = False  # False if it truncates.
    ARG_SYMBOL = None   # Not implemented by Dremio
    TYPE_CLASSES = {
        # https://docs.dremio.com/current/reference/sql/data-types/
        # com.dremio.dac.explore.DataTypeUtil
        "BOOLEAN": Boolean,
        "INT": Integer,
        "INTEGER": Integer,
        "TINYINT": Integer,  # An alias for Integer.
        "SMALLINT": Integer,  # An alias for Integer.
        "BIGINT": Integer,
        "DECIMAL": Decimal,
        "DEC": Decimal,  # An alias for Decimal.
        "NUMERIC": Decimal,  # An alias for Decimal.
        "FLOAT": Float,
        "DOUBLE": Float,
        "REAL": Float,
        "DATE": Date,
        "TIME": Timestamp,
        "TIMESTAMP": Timestamp,  # Dremio does not support timezone-aware timestamps.
        "CHAR": Text,
        "CHARACTER": Text,
        "VARCHAR": Text,
        "CHARACTER VARYING": Text,  # An alias for VARCHAR.
        "VARBINARY": Text,
        "BINARY VARYING": Text,  # An alias for VARBINARY.
    }
    MIXINS = {Mixin_Schema, Mixin_MD5, Mixin_NormalizeValue, Mixin_RandomSample}

    def list_tables(self, table_schema: str, like: Compilable = None) -> Compilable:
        """Query to select the list of tables in the schema. (query return type: table[str])

         If 'like' is specified, the value is applied to the table name, using the 'like' operator.
         """
        return (
            self.table_information()
            .where(
                this.table_schema == table_schema,
                this.table_name.like(like) if like is not None else SKIP,
            )
            .select(this.table_name)
        )

    def timestamp_value(self, t: DbTime) -> str:
        s = t.isoformat(' ', 'milliseconds')
        return f"cast('{s}' as timestamp)"

    def quote(self, s: str):
        return f'"{s}"'

    def to_string(self, s: str):
        return f"cast({s} as varchar)"

    @staticmethod
    def nan_to_none(value: Optional[object]) -> Optional[int]:
        from math import isnan
        if value is not None and not isinstance(value, int):
            return None if isnan(value) else int(value)
        else:
            return value

    def parse_type(
        self,
        table_path: DbPath,
        col_name: str,
        type_repr: str,
        datetime_precision: int = None,
        numeric_precision: int = None,
        numeric_scale: int = None,
    ) -> ColType:
        datetime_precision = self.nan_to_none(datetime_precision)
        numeric_precision = self.nan_to_none(numeric_precision)
        numeric_scale = self.nan_to_none(numeric_scale)

        return super().parse_type(table_path, col_name, type_repr, datetime_precision, numeric_precision, numeric_scale)

    def set_timezone_to_utc(self) -> str:
        raise NotImplementedError(
            'Dremio does not support setting session timezone to UTC. Dremio retrieves the time or timestamp value '
            'with the assumption that the time zone is in Coordinated Universal Time (UTC). Use CONVERT_TIMEZONE in '
            'queries instead.'
        )


class Dremio(ThreadedDatabase):
    dialect = Dialect()
    CONNECT_URI_HELP = ("dremio://<user>[:<password>]@<host>:<port>/<space>?"
                        "Token=<Token>&"
                        "UseEncryption=<UseEncryption>&"
                        "DisableCertificateVerification=<DisableCertificateVerification>")
    CONNECT_URI_PARAMS = ['space?', 'Token?', 'UseEncryption?', 'DisableCertificateVerification?']

    def __init__(self, thread_count, **kw):
        self._args = kw
        if 'space' in self._args:
            self.default_schema = self._args['space']
        super().__init__(thread_count=thread_count)

    def create_connection(self):
        # Dremio PAT token is preferred over password. It must be URL-encoded as it will contain special characters.
        password_or_token = self._args.get('Token') if self._args.get('Token') else self._args.get('password')
        user_pw_str = f"{self._args.get('user')}:{password_or_token}@"
        keywords = []
        if 'UseEncryption' in self._args:
            keywords.append(f"UseEncryption={self._args['UseEncryption']}")
        if 'DisableCertificateVerification' in self._args:
            keywords.append(f"DisableCertificateVerification={self._args['DisableCertificateVerification']}")
        keyword_str = '?' + '&'.join(keywords) if keywords else ''

        connection_string = (f"dremio+flight://{user_pw_str}{self._args['host']}:{self._args['port']}/"
                             f"{self.default_schema}{keyword_str}")

        dremiodb = import_dremio()
        engine: dremiodb.Engine = dremiodb.create_engine(url=connection_string, echo=True)

        try:
            connection: dremiodb.engine.Connection = engine.connect()
        except dremiodb.exc.SQLAlchemyError as e:
            raise ConnectionError(*e.args) from e

        return connection.connection

    def _query(self, sql_code: Union[str, ThreadLocalInterpreter]):
        "Uses the standard SQL cursor interface"
        from pyarrow._flight import FlightUnauthenticatedError
        try:
            return super()._query(sql_code=sql_code)
        except FlightUnauthenticatedError as e:
            raise ValueError(
                'Possible reasons - Your TOKEN may have expired in the Dremio -> Account Settings -> Personal Access '
                'Token page. Or you have not provided a username that matches the Personal Access Token: {e}'
            ) from e

    def parse_table_name(self, name: str) -> DbPath:
        path = parse_table_name(name)
        return tuple(i for i in self._normalize_table_path(path) if i is not None)

    @property
    def is_autocommit(self) -> bool:
        return True
