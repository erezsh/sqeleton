import unittest
import tempfile
import os
from datetime import datetime
from typing import Callable, List, Tuple

import pytz

from sqeleton import connect
from sqeleton import databases as dbs
from sqeleton.queries import table, current_timestamp, NormalizeAsString, ForeignKey, Compiler
from .common import str_to_checksum, make_test_each_database_in_list, get_conn, random_table_suffix
from sqeleton.abcs.database_types import TimestampTZ, Timestamp, Decimal
from sqeleton.databases.base import TIMESTAMP_PRECISION_POS
from sqeleton.abcs.mixins import AbstractMixin_MD5

TEST_DATABASES = {
    dbs.MySQL,
    dbs.PostgreSQL,
    dbs.Oracle,
    dbs.DuckDB,
    dbs.Presto,
    dbs.Trino,
    dbs.Dremio,
    dbs.BigQuery,
    dbs.Snowflake,
    dbs.Redshift,
    dbs.Vertica,
}

test_each_database: Callable = make_test_each_database_in_list(TEST_DATABASES)


@test_each_database
class TestDatabase(unittest.TestCase):
    def test_connect_to_db(self):
        db = get_conn(self.db_cls)
        self.assertEqual(1, db.query("SELECT 1", int))


@test_each_database
class TestMD5(unittest.TestCase):
    def test_md5_as_int(self):
        db = get_conn(self.db_cls)
        
        # Check if the database dialect has Mixin_MD5 in its MIXINS

        has_md5_mixin = any(issubclass(mixin, AbstractMixin_MD5) for mixin in db.dialect.MIXINS)
        
        if not has_md5_mixin:
            self.skipTest(f"{self.db_cls.__name__} does not support MD5")
        
        # Load the MD5 mixin into the dialect
        dialect_with_md5 = db.dialect.load_mixins(AbstractMixin_MD5)
        
        str_value = "hello world"
        query_fragment = dialect_with_md5.md5_as_int("'{0}'".format(str_value))
        query = f"SELECT {query_fragment}"

        self.assertEqual(str_to_checksum(str_value), db.query(query, int))


class TestConnect(unittest.TestCase):
    def test_bad_uris(self):
        self.assertRaises(ValueError, connect, "p")
        self.assertRaises(ValueError, connect, "postgresql:///bla/foo")
        self.assertRaises(ValueError, connect, "snowflake://user:pass@foo/bar/TEST1")
        self.assertRaises(ValueError, connect, "snowflake://user:pass@foo/bar/TEST1?warehouse=ha&schema=dup")


@test_each_database
class TestSchema(unittest.TestCase):
    def test_table_list(self):
        name = "tbl_" + random_table_suffix()
        db = get_conn(self.db_cls)
        tbl = table(db.parse_table_name(name), schema={"id": int})
        q = db.dialect.list_tables(db.default_schema, name)
        assert not db.query(q)

        db.query(tbl.create())
        self.assertEqual(db.query(q, List[str]), [name])

        db.query(tbl.drop())
        assert not db.query(q)

    def test_type_mapping(self):
        name = "tbl_" + random_table_suffix()
        db = get_conn(self.db_cls)
        tbl = table(
            db.parse_table_name(name),
            schema={
                "int": int,
                "float": float,
                "datetime": datetime,
                "str": str,
                "bool": bool,
            },
        )
        q = db.dialect.list_tables(db.default_schema, name)
        assert not db.query(q)

        db.query(tbl.create())
        self.assertEqual(db.query(q, List[str]), [name])

        db.query(tbl.drop())
        assert not db.query(q)


@test_each_database
class TestQueries(unittest.TestCase):
    def test_current_timestamp(self):
        db = get_conn(self.db_cls)
        res = db.query(current_timestamp(), datetime)
        assert isinstance(res, datetime), (res, type(res))

    def test_correct_timezone(self):
        name = "tbl_" + random_table_suffix()
        db = get_conn(self.db_cls)
        tbl = table(name, schema={"id": int, "created_at": TimestampTZ(9), "updated_at": TimestampTZ(9)})

        db.query(tbl.create())

        tz = pytz.timezone("UTC")

        now = datetime.now(tz)
        if isinstance(db, (dbs.Presto, dbs.Trino, dbs.Dremio)):
            ms = now.microsecond // 1000 * 1000  # Presto/Trino max precision is 3
            now = now.replace(microsecond=ms)

        db.query(tbl.insert_row(1, now, now))
        if self.db_cls not in [dbs.Dremio, dbs.Presto]:
            db.query(db.dialect.set_timezone_to_utc())

        t = db.table(tbl).query_schema()
        t.schema["created_at"] = t.schema["created_at"].replace(precision=t.schema["created_at"].precision)

        tbl = table(name, schema=t.schema)

        results = db.query(tbl.select(NormalizeAsString(tbl[c]) for c in ["created_at", "updated_at"]), List[Tuple])

        created_at = results[0][1]
        updated_at = results[0][1]

        utc = now.astimezone(pytz.UTC)
        expected = utc.__format__("%Y-%m-%d %H:%M:%S.%f")

        self.assertEqual(created_at, expected)
        self.assertEqual(updated_at, expected)

        db.query(tbl.drop())

    def test_foreign_key(self):
        db = get_conn(self.db_cls)

        a = table("tbl1_" + random_table_suffix(), schema={"id": int})
        b = table("tbl2_" + random_table_suffix(), schema={"a": ForeignKey(a, "id")})
        c = Compiler(db)

        s = c.compile(b.create())
        try:
            db.query([a.create(), b.create()])

            print("TODO foreign key")
            # breakpoint()
        finally:
            db.query(
                [
                    a.drop(True),
                    b.drop(True),
                ]
            )


@test_each_database
class TestNormalizeValue(unittest.TestCase):
    def test_normalize_uuid(self):
        """Test UUID normalization by creating a table with UUID column and querying it"""
        db = get_conn(self.db_cls)

        # Skip if database doesn't have normalize_uuid
        if not hasattr(db.dialect, 'normalize_uuid'):
            self.skipTest(f"{self.db_cls.__name__} does not support UUID normalization")
        
        # Skip for Presto if connected to read-only catalog
        # if self.db_cls == dbs.Presto:
        #     self.skipTest(f"Presto may not support table creation in current catalog")

        table_name = "tbl_" + random_table_suffix()

        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        insert_value = f" {test_uuid} "
        schema = {"id": int, "uuid_col": str}
        tbl = table(db.parse_table_name(table_name), schema=schema)

        try:
            db.query(tbl.create())
            # Insert the UUID value with extra spaces to see if normalization trims it
            db.query(tbl.insert_row(1, insert_value))

            # Query the schema to get proper column types with refinement
            t = db.table(*tbl.path).query_schema()
            col_type = t.schema["uuid_col"]

            # Build a normalized query
            normalized_expr = db.dialect.normalize_uuid(db.dialect.quote("uuid_col"), col_type)
            result = db.query(f"SELECT {normalized_expr} FROM {db.dialect.quote(table_name)}", str)

            # Should return the UUID
            self.assertEqual(result, test_uuid)

        finally:
            db.query(tbl.drop())
    
    def test_normalize_timestamp(self):
        """Test timestamp normalization by creating a table and querying with different precisions"""
        db = get_conn(self.db_cls)
        
        if not hasattr(db.dialect, 'normalize_timestamp'):
            self.skipTest(f"{self.db_cls.__name__} does not support timestamp normalization")
        
        # Skip for Presto if connected to read-only catalog
        # if self.db_cls == dbs.Presto:
        #     self.skipTest(f"Presto may not support table creation in current catalog")
        
        table_name = "tbl_" + random_table_suffix()
        tbl = table(db.parse_table_name(table_name), schema={"id": int, "ts_col": datetime})
        
        try:
            db.query(tbl.create())
            
            # Insert a timestamp with microseconds
            test_ts = datetime(2023, 6, 15, 14, 30, 45, 123456)
            db.query(tbl.insert_row(1, test_ts))
            
            # Query the schema to get proper column types
            t = db.table(*tbl.path).query_schema()
            col_type = t.schema["ts_col"]
            
            # Test with different precisions
            for precision in [0, 3, 6]:
                test_col_type = Timestamp(precision=precision, rounds=col_type.rounds)
                normalized_expr = db.dialect.normalize_timestamp(db.dialect.quote("ts_col"), test_col_type)
                result = db.query(f"SELECT {normalized_expr} FROM {db.dialect.quote(table_name)}", str)
                
                # Verify the result is a properly formatted timestamp string
                self.assertIsInstance(result, str)
                self.assertTrue(result.startswith("2023-06-15"))
                # Should have the expected length (YYYY-MM-DD HH:MI:SS.FFFFFF)
                self.assertEqual(len(result), TIMESTAMP_PRECISION_POS + 6)
                
        finally:
            db.query(tbl.drop())
    
    def test_normalize_number(self):
        """Test number normalization by creating a table with decimal columns"""
        db = get_conn(self.db_cls)
        
        if not hasattr(db.dialect, 'normalize_number'):
            self.skipTest(f"{self.db_cls.__name__} does not support number normalization")
        
        # Skip for Presto if connected to read-only catalog
        # if self.db_cls == dbs.Presto:
        #     self.skipTest(f"Presto may not support table creation in current catalog")
        
        table_name = "tbl_" + random_table_suffix()
        tbl = table(db.parse_table_name(table_name), schema={"id": int, "num_col": float})
        
        try:
            db.query(tbl.create())
            
            # Insert various numeric values
            test_values = [
                (1, 123.456),
                (2, 0.123),
                (3, 999.999),
                (4, 0.0),
            ]
            
            for id_val, num_val in test_values:
                db.query(tbl.insert_row(id_val, num_val))
            
            # Query the schema to get proper column types
            t = db.table(*tbl.path).query_schema()
            col_type = t.schema["num_col"]
            
            # Test with different precisions
            for precision in [0, 2, 6]:
                test_col_type = Decimal(precision=precision)
                normalized_expr = db.dialect.normalize_number(db.dialect.quote("num_col"), test_col_type)
                # Use the dialect's quote method for the ORDER BY column as well
                order_by_col = db.dialect.quote("id")
                results = db.query(
                    f"SELECT {normalized_expr} FROM {db.dialect.quote(table_name)} ORDER BY {order_by_col}",
                    List[str]
                )
                
                # Verify all results are strings (normalized)
                for result in results:
                    self.assertIsInstance(result, str)
                    # Should not have leading/trailing spaces
                    self.assertEqual(result, result.strip())
                    
        finally:
            db.query(tbl.drop())


@test_each_database
class TestThreePartIds(unittest.TestCase):
    def test_three_part_support(self):
        # DuckDB doesn't support querying information_schema.columns with three-part identifiers
        # for attached databases. Its information_schema is database-scoped, not catalog-scoped.
        if self.db_cls not in [dbs.PostgreSQL, dbs.Redshift, dbs.Snowflake]:
            self.skipTest("Limited support for 3 part ids")

        table_name = "tbl_" + random_table_suffix()
        db = get_conn(self.db_cls)
        db_res = db.query("SELECT CURRENT_DATABASE()")
        schema_res = db.query("SELECT CURRENT_SCHEMA()")
        db_name = db_res.rows[0][0]
        schema_name = schema_res.rows[0][0]

        table_one_part = table((table_name,), schema={"id": int})
        table_two_part = table((schema_name, table_name), schema={"id": int})
        table_three_part = table((db_name, schema_name, table_name), schema={"id": int})

        for part in (table_one_part, table_two_part, table_three_part):
            db.query(part.create())
            d = db.query_table_schema(part.path)
            assert len(d) == 1
            db.query(part.drop())
