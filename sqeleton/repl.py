import rich.table
import logging

from pathlib import Path
from time import time

### XXX Fix for Python 3.8 bug (https://github.com/prompt-toolkit/python-prompt-toolkit/issues/1023)
import asyncio
import selectors

selector = selectors.SelectSelector()
loop = asyncio.SelectorEventLoop(selector)
asyncio.set_event_loop(loop)
### XXX End of fix

from pygments.lexers.sql import SqlLexer
from pygments.styles import get_style_by_name
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.filters import Condition
from prompt_toolkit.application.current import get_app
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles.pygments import style_from_pygments_cls

from . import __version__

STYLE = style_from_pygments_cls(get_style_by_name("dracula"))


sql_keywords = [
    "abort",
    "action",
    "add",
    "after",
    "all",
    "alter",
    "analyze",
    "and",
    "as",
    "asc",
    "attach",
    "autoincrement",
    "before",
    "begin",
    "between",
    "by",
    "cascade",
    "case",
    "cast",
    "check",
    "collate",
    "column",
    "commit",
    "conflict",
    "constraint",
    "create",
    "cross",
    "current_date",
    "current_time",
    "current_timestamp",
    "database",
    "default",
    "deferrable",
    "deferred",
    "delete",
    "desc",
    "detach",
    "distinct",
    "drop",
    "each",
    "else",
    "end",
    "escape",
    "except",
    "exclusive",
    "exists",
    "explain",
    "fail",
    "for",
    "foreign",
    "from",
    "full",
    "glob",
    "group",
    "having",
    "if",
    "ignore",
    "immediate",
    "in",
    "index",
    "indexed",
    "initially",
    "inner",
    "insert",
    "instead",
    "intersect",
    "into",
    "is",
    "isnull",
    "join",
    "key",
    "left",
    "like",
    "limit",
    "match",
    "natural",
    "no",
    "not",
    "notnull",
    "null",
    "of",
    "offset",
    "on",
    "or",
    "order",
    "outer",
    "plan",
    "pragma",
    "primary",
    "query",
    "raise",
    "recursive",
    "references",
    "regexp",
    "reindex",
    "release",
    "rename",
    "replace",
    "restrict",
    "right",
    "rollback",
    "row",
    "savepoint",
    "select",
    "set",
    "table",
    "temp",
    "temporary",
    "then",
    "to",
    "transaction",
    "trigger",
    "union",
    "unique",
    "update",
    "using",
    "vacuum",
    "values",
    "view",
    "virtual",
    "when",
    "where",
    "with",
    "without",
]
sql_completer = WordCompleter(sql_keywords, ignore_case=True)


def add_keywords(new_keywords):
    global sql_keywords
    new = set(new_keywords) - set(sql_keywords)
    sql_keywords += new


def _code_is_valid(code: str):
    if code:
        if code[0].isalnum():
            return code[-1] in ";\n"
    return True


def repl(uri, prompt=" >> "):
    rich.print(f"[purple]Sqeleton {__version__} interactive prompt. Enter '?' for help[/purple]")

    db = connect(uri)
    db_name = db.name

    try:
        session = PromptSession(
            lexer=PygmentsLexer(SqlLexer),
            completer=sql_completer,
            # key_bindings=kb
            history=FileHistory(str(Path.home() / ".sqeleton_repl_history")),
            auto_suggest=AutoSuggestFromHistory(),
            color_depth=ColorDepth.TRUE_COLOR,
            style=STYLE,
        )

        @Condition
        def multiline_filter():
            text = get_app().layout.get_buffer_by_name("DEFAULT_BUFFER").text
            return not _code_is_valid(text)

        prompt = f"{db_name}> "
        while True:
            # Read
            try:
                q = session.prompt(prompt, multiline=multiline_filter)
                if not q.strip():
                    continue

                start_time = time()
                run_command(db, q)

                duration = time() - start_time
                if duration > 1:
                    rich.print("(Query took %.2f seconds)" % duration)

            except KeyboardInterrupt:
                rich.print("Interrupted (Ctrl+C)")

    except (KeyboardInterrupt, EOFError):
        rich.print("Exiting Sqeleton interaction")


from . import connect

import sys


def print_table(rows, schema, table_name=""):
    # Print rows in a rich table
    t = rich.table.Table(title=table_name, caption=f"{len(rows)} rows")
    for col in schema:
        t.add_column(col)
    for r in rows:
        t.add_row(*map(str, r))
    rich.print(t)


def help():
    rich.print("Commands:")
    rich.print("  ?mytable - shows schema of table 'mytable'")
    rich.print("  * - shows list of all tables")
    rich.print("  *pattern - shows list of all tables with name like pattern")
    rich.print("Otherwise, runs regular SQL query")


def run_command(db, q):
    if q.startswith("*"):
        pattern = q[1:]
        try:
            names = db.query(db.dialect.list_tables(db.default_schema, like=f"%{pattern}%" if pattern else None))
        except Exception as e:
            logging.exception(e)
        else:
            print_table(names, ["name"], "List of tables")
            add_keywords([".".join(n) for n in names])
    elif q.startswith("?"):
        table_name = q[1:]
        if not table_name:
            help()
            return
        try:
            path = db.parse_table_name(table_name)
            print("->", path)
            schema = db.query_table_schema(path)
        except Exception as e:
            logging.error(e)
        else:
            print_table([(k, v[1]) for k, v in schema.items()], ["name", "type"], f"Table '{table_name}'")
            add_keywords(schema.keys())
    else:
        # Normal SQL query
        try:
            res = db.query(q)
        except Exception as e:
            logging.error(e)
        else:
            if res:
                print_table(res.rows, res.columns, None)
                add_keywords(res.columns)


def main():
    uri = sys.argv[1]
    return repl(uri)


if __name__ == "__main__":
    main()
