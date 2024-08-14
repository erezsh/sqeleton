from ..abcs.database_types import TemporalType, ColType_UUID, String_UUID
from . import presto
from .base import import_helper
from .base import TIMESTAMP_PRECISION_POS


@import_helper("trino")
def import_trino():
    import trino

    return trino


Mixin_MD5 = presto.Mixin_MD5


class Mixin_NormalizeValue(presto.Mixin_NormalizeValue):
    def normalize_timestamp(self, value: str, coltype: TemporalType) -> str:
        if coltype.rounds:
            s = f"date_format(cast({value} as timestamp({coltype.precision})), '%Y-%m-%d %H:%i:%S.%f')"
        else:
            s = f"date_format(cast({value} as timestamp(6)), '%Y-%m-%d %H:%i:%S.%f')"

        return f"RPAD(RPAD({s}, {TIMESTAMP_PRECISION_POS + coltype.precision}, '.'), {TIMESTAMP_PRECISION_POS + 6}, '0')"

    def normalize_uuid(self, value: str, coltype: ColType_UUID) -> str:
        if isinstance(coltype, String_UUID):
            return f"TRIM({value})"
        return f"CAST({value} AS VARCHAR)"


class Dialect(presto.Dialect):
    name = "Trino"
    ARG_SYMBOL = "?"

    def set_timezone_to_utc(self) -> str:
        return "SET TIME ZONE '+00:00'"


class Trino(presto.Presto):
    dialect = Dialect()
    CONNECT_URI_HELP = "trino://<user>@<host>/<catalog>/<schema>"
    CONNECT_URI_PARAMS = ["catalog", "schema"]

    def __init__(self, **kw):

        trino = import_trino()

        if kw.get("schema"):
            self.default_schema = kw.get("schema")

        if kw.get("password"):
            kw["auth"] = trino.auth.BasicAuthentication(
                kw.pop("user"), kw.pop("password")
            )
            kw["http_scheme"] = "https"

        cert = kw.pop("cert", None)
        self._conn = trino.dbapi.connect(**kw)
        if cert is not None:
            self._conn._http_session.verify = cert


    @property
    def is_autocommit(self) -> bool:
        return True