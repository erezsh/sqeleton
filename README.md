# Sqeleton

Sqeleton is a Python library for querying SQL databases.

It consists of -

- A fast and concise query builder, designed from scratch, but inspired by PyPika and SQLAlchemy

- A modular database interface, with drivers for a long list of SQL databases.

It is comparable to other libraries such as SQLAlchemy or PyPika, in terms of API and intended audience. However, there are several notable ways in which it is different. 

## **Features:**

🏃‍♂️**High-performance**: Sqeleton's API is designed to maximize performance using batch operations

- No ORM! While ORMs are easy and familiar, their granular operations are far too slow.
- Compiles queries 4 times faster than SQLAlchemy

🙌**Parallel**: Seamless multi-threading and multi-processing support

💖**Well-tested**: In addition to having an extensive test-suite, sqeleton is used as the core of [data-diff](https://github.com/datafold/data-diff).

✅**Type-aware**: The schema is used for validation when building expressions, making sure the names are correct, and that the data-types align. (WIP)
    
- The schema can be queried at run-time, if the tables already exist in the database

✨**Multi-database access**: Sqeleton is designed to work with several databases at the same time. Its API abstracts away as many implementation details as possible.

_Databases we fully support_:

- PostgreSQL >=10
- MySQL
- Snowflake
- BigQuery
- Redshift
- Oracle
- Presto
- Databricks
- Trino
- Clickhouse
- Vertica
- DuckDB >=0.6
- SQLite (coming soon)

💻**Built-in SQL client**: Connect to any of the supported databases with just one line.

Example usage: `sqeleton repl snowflake://...`

- Has syntax-highlighting, and autocomplete
- Use `*text` to find all tables like `%text%` (or just `*` to see all tables)
- Use `?name` to see the schema of the table called `name`.

## Documentation

[Read the docs!](https://sqeleton.readthedocs.io)

Or jump straight to the [introduction](https://sqeleton.readthedocs.io/en/latest/intro.html).

### Install

Install using pip:

```bash
pip install sqeleton
```

It is recommended to install the driver dependencies using pip's `[]` syntax:

```bash
pip install 'sqeleton[mysql, postgresql]'
```

Read more in [install / getting started.](https://sqeleton.readthedocs.io/en/latest/install.html)

### Example: Basic usage

We will create a table with the numbers 0..100, and then sum them up.

```python
from sqeleton import connect, table, this

# Create a new database connection
ddb = connect("duckdb://:memory:")

# Define a table with one int column
tbl = table('my_list', schema={'item': int})

# Make a bunch of queries
queries = [
    # Create table 'my_list'
    tbl.create(),

    # Insert 100 numbers
    tbl.insert_rows([x] for x in range(100)),

    # Get the sum of the numbers
    tbl.select(this.item.sum())
]
# Query in order, and return the last result as an int
result = ddb.query(queries, int)    

# Prints: Total sum of 0..100 = 4950
print(f"Total sum of 0..100 = {result}")
```

### Example: Advanced usage

We will define a function that performs outer-join on any database, and adds two extra fields: `only_a` and `only_b`.

```python
from sqeleton.databases import Database
from sqeleton.queries import ITable, leftjoin, rightjoin, outerjoin, and_, Expr

def my_outerjoin(
        db: Database,
        a: ITable, b: ITable,
        keys1: List[str], keys2: List[str],
        select_fields: Dict[str, Expr]
    ) -> ITable:
    """This function accepts two table expressions, and returns an outer-join query.
    
    The resulting rows will include two extra boolean fields:
    "only_a", and "only_b", describing whether there was a match for that row 
    only in the first table, or only in the second table.

    Parameters:
        db - the database connection to use
        a, b - the tables to outer-join
        keys1, keys2 - the names of the columns to join on, for each table respectively
        select_fields - A dictionary of {column_name: expression} to select as a result of the outer-join
    """
    # Predicates to join on
    on = [a[k1] == b[k2] for k1, k2 in zip(keys1, keys2)]

    # Define the new boolean fields
    # If all keys are None, it means there was no match
    # Compiles to "<k1> IS NULL AND <k2> IS NULL AND <k3> IS NULL..." etc.
    only_a = and_(b[k] == None for k in keys2)
    only_b = and_(a[k] == None for k in keys1)

    if isinstance(db, MySQL):
        # MySQL doesn't support "outer join"
        # Instead, we union "left join" and "right join"
        l = leftjoin(a, b).on(*on).select(
                only_a=only_a,
                only_b=False,
                **select_fields
            )
        r = rightjoin(a, b).on(*on).select(
                only_a=False,
                only_b=only_b,
                **select_fields
            )
        return l.union(r)

    # Other databases
    return outerjoin(a, b).on(*on).select(
            only_a=only_a,
            only_b=only_b,
            **select_fields
        )
```



# TODO

- Transactions

- Indexes

- Date/time expressions

- Window functions

## Possible plans for the future (not determined yet)

- Cache the compilation of repetitive queries for even faster query-building

- Compile control flow, functions

- Define tables using type-annotated classes (SQLModel style)

## Alternatives

- [SQLAlchemy](https://www.sqlalchemy.org/)
- [PyPika](https://github.com/kayak/pypika)
- [PonyORM](https://ponyorm.org/)
- [peewee](https://github.com/coleifer/peewee)

# Thanks

Thanks to Datafold for having sponsored Sqeleton in its initial stages. For reference, [the original repo](https://github.com/datafold/sqeleton/).