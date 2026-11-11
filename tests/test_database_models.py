"""
Tests for src/database_models.py (ORM models, duplicate detection, voting,
statistics, init_db / get_db and DATABASE_URL handling).

All tests run against the temporary SQLite database set up by conftest.py
(``db_session`` = fresh empty schema). The real ``university_bot.db`` is only
ever *copied* (read-only) for the schema-compatibility test.
"""

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError

import database_models as dm
from database_models import (
    RawData,
    SystemMetrics,
    TrainingData,
    UserAnalytics,
    UserVotes,
)

ROOT = Path(__file__).resolve().parent.parent
REAL_DB = ROOT / "university_bot.db"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def add_question(db, question, telegram_id=1, **kwargs):
    row = RawData(telegram_id=telegram_id, question=question, **kwargs)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def add_training(db, question, answer, **kwargs):
    row = TrainingData(question=question, answer=answer, **kwargs)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def reload(db, model, row_id):
    db.expire_all()
    return db.get(model, row_id)


def sklearn_reference_score(t1, t2):
    """The previous implementation's TF-IDF scoring (sklearn, 3 documents)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 1), min_df=1,
                          token_pattern=r"[^\W_]+")
    m = vec.fit_transform([dm.clean_text(t1), dm.clean_text(t2), "dummy document text"])
    return float(cosine_similarity(m[0:1], m[1:2])[0][0])


@pytest.fixture
def other_session():
    """A second, independent session on the same temporary database."""
    sessions = []

    def factory():
        s = dm.SessionLocal()
        sessions.append(s)
        return s

    yield factory
    for s in sessions:
        s.rollback()
        s.close()


# --------------------------------------------------------------------------- #
# DATABASE_URL handling (decision d)
# --------------------------------------------------------------------------- #

class TestDatabaseUrl:
    def test_default_is_project_root_db(self):
        expected = f"sqlite:///{os.path.join(dm.BASE_DIR, 'university_bot.db')}"
        assert dm.DEFAULT_DATABASE_URL == expected
        assert Path(dm.BASE_DIR) == ROOT
        for empty in (None, "", "   "):
            assert dm.resolve_database_url(empty) == expected

    def test_relative_sqlite_resolved_against_project_root(self):
        assert dm.resolve_database_url("sqlite:///university_bot.db") == dm.DEFAULT_DATABASE_URL
        assert dm.resolve_database_url("sqlite:///./data/../x/test.db") == \
            f"sqlite:///{ROOT / 'x' / 'test.db'}"

    def test_absolute_memory_and_other_backends_untouched(self, tmp_path):
        absolute = f"sqlite:///{tmp_path / 'a.db'}"
        assert dm.resolve_database_url(absolute) == absolute
        assert dm.resolve_database_url("sqlite://") == "sqlite://"
        assert dm.resolve_database_url("sqlite:///:memory:") == "sqlite:///:memory:"
        pg = "postgresql://user:secret@db.example:5432/cengbot"
        assert dm.resolve_database_url(pg) == pg

    def test_env_var_wins_over_dotenv_file(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text("DATABASE_URL=sqlite:///from_dotenv.db\n")
        monkeypatch.setattr(dm, "BASE_DIR", str(tmp_path))
        monkeypatch.setenv("DATABASE_URL", "sqlite:///from_env.db")
        assert dm._read_env_database_url() == "sqlite:///from_env.db"

    def test_dotenv_file_used_when_env_var_missing(self, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text("DATABASE_URL=sqlite:///from_dotenv.db\n")
        monkeypatch.setattr(dm, "BASE_DIR", str(tmp_path))
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert dm._read_env_database_url() == "sqlite:///from_dotenv.db"
        # os.environ is not modified by the fallback
        assert "DATABASE_URL" not in os.environ

    def test_no_env_and_no_dotenv_gives_none(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dm, "BASE_DIR", str(tmp_path))
        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert dm._read_env_database_url() is None

    def test_conftest_database_is_used(self, db_session):
        # module was imported after conftest set DATABASE_URL -> temp DB
        assert "cengbot_tests_" in dm.DATABASE_URL
        assert "cengbot_tests_" in str(db_session.get_bind().url)

    def test_import_in_fresh_process(self, tmp_path):
        """Relative DATABASE_URL resolves against the root even from another
        cwd, and importing the module does not configure the root logger."""
        code = (
            "import json, logging, sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "import database_models as dm\n"
            "print(json.dumps({'url': dm.DATABASE_URL,\n"
            "                  'engine': dm.engine.url.render_as_string(hide_password=False),\n"
            "                  'root_handlers': len(logging.getLogger().handlers)}))\n"
        )
        env = dict(os.environ)
        env["DATABASE_URL"] = "sqlite:///_agent1_never_created_dir/test.db"
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(tmp_path), env=env,
            capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        info = json.loads(out.stdout.strip().splitlines()[-1])
        expected = f"sqlite:///{ROOT / '_agent1_never_created_dir' / 'test.db'}"
        assert info["url"] == expected
        assert info["engine"] == expected
        assert info["root_handlers"] == 0
        # creating the engine must not create anything on disk
        assert not (ROOT / "_agent1_never_created_dir").exists()


# --------------------------------------------------------------------------- #
# Models / schema
# --------------------------------------------------------------------------- #

class TestModels:
    def test_raw_data_defaults(self, db_session):
        row = add_question(db_session, "Staj başvurusu ne zaman?")
        assert row.admin_approved == 0
        assert row.is_duplicate is False
        assert row.like is None
        assert isinstance(row.created_at, datetime)
        assert row.created_at.tzinfo is None  # stored as naive UTC like existing data
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        assert abs(row.created_at - utc_now) < timedelta(minutes=1)
        assert row.last_updated is not None

    def test_last_updated_changes_on_update(self, db_session):
        row = add_question(db_session, "Soru")
        old = datetime(2000, 1, 1)
        db_session.execute(text("UPDATE raw_data SET last_updated = :t WHERE id = :i"),
                           {"t": old, "i": row.id})
        db_session.commit()
        row = reload(db_session, RawData, row.id)
        row.answer = "Cevap"
        db_session.commit()
        row = reload(db_session, RawData, row.id)
        assert row.last_updated > old
        assert row.created_at > old  # created_at untouched by the update

    def test_training_data_defaults_and_relationship(self, db_session):
        raw = add_question(db_session, "Q", answer="A")
        td = add_training(db_session, "Q", "A", source_id=raw.id)
        assert td.is_answer_duplicate is False
        assert td.usage_count == 0
        assert td.training_weight == 1.0
        assert td.review_status == "pending"
        assert td.is_active is True
        assert td.version == 1
        assert td.source.id == raw.id
        assert [t.id for t in reload(db_session, RawData, raw.id).training_data] == [td.id]

    def test_training_source_id_unique(self, db_session):
        raw = add_question(db_session, "Q", answer="A")
        add_training(db_session, "Q", "A", source_id=raw.id)
        db_session.add(TrainingData(question="Q", answer="A2", source_id=raw.id))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_user_votes_unique_per_user_and_question(self, db_session):
        raw = add_question(db_session, "Q")
        db_session.add(UserVotes(raw_data_id=raw.id, telegram_user_id=7, current_vote=1))
        db_session.commit()
        db_session.add(UserVotes(raw_data_id=raw.id, telegram_user_id=7, current_vote=-1))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()
        # other user / other question is fine
        db_session.add(UserVotes(raw_data_id=raw.id, telegram_user_id=8, current_vote=-1))
        db_session.commit()

    def test_other_models_defaults(self, db_session):
        ua = UserAnalytics(telegram_id=5)
        sm = SystemMetrics(metric_name="cpu", metric_value=1.5)
        db_session.add_all([ua, sm])
        db_session.commit()
        assert ua.total_questions == 0 and ua.created_at is not None
        assert sm.timestamp is not None

    def test_model_indexes_present_on_fresh_schema(self, test_engine, db_session):
        insp = inspect(test_engine)
        vote_indexes = {i["name"]: i for i in insp.get_indexes("user_votes")}
        assert vote_indexes["idx_user_votes_unique"]["unique"] == 1
        assert vote_indexes["idx_user_votes_unique"]["column_names"] == ["raw_data_id", "telegram_user_id"]
        raw_indexes = {i["name"] for i in insp.get_indexes("raw_data")}
        assert {"idx_raw_data_created_at", "idx_raw_data_composite"} <= raw_indexes


@pytest.mark.skipif(not REAL_DB.exists(), reason="real university_bot.db not present")
class TestRealSchemaCompatibility:
    """The models must match the production schema (no migration needed)."""

    @pytest.fixture
    def real_copy_engine(self, tmp_path):
        copy = tmp_path / "real_copy.db"
        shutil.copyfile(REAL_DB, copy)  # read-only access to the real file
        eng = create_engine(f"sqlite:///{copy}")
        yield eng
        eng.dispose()

    def test_columns_match_real_db(self, real_copy_engine):
        insp = inspect(real_copy_engine)
        for table in dm.Base.metadata.sorted_tables:
            assert insp.has_table(table.name), f"table {table.name} missing in real DB"
            real_cols = {c["name"]: c for c in insp.get_columns(table.name)}
            model_cols = {c.name: c for c in table.columns}
            assert set(model_cols) == set(real_cols), table.name
            for name, col in model_cols.items():
                if not col.primary_key:
                    assert bool(real_cols[name]["nullable"]) == bool(col.nullable), (table.name, name)

    def test_init_db_is_noop_on_real_schema(self, real_copy_engine, monkeypatch, capsys):
        before = {t: {i["name"] for i in inspect(real_copy_engine).get_indexes(t)}
                  for t in inspect(real_copy_engine).get_table_names()}
        monkeypatch.setattr(dm, "engine", real_copy_engine)
        dm.init_db()
        insp = inspect(real_copy_engine)
        after = {t: {i["name"] for i in insp.get_indexes(t)} for t in insp.get_table_names()}
        assert after == before


# --------------------------------------------------------------------------- #
# init_db / get_db
# --------------------------------------------------------------------------- #

class TestInitDbAndGetDb:
    def test_init_db_uses_module_engine_at_call_time(self, tmp_path, monkeypatch, capsys):
        eng = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
        monkeypatch.setattr(dm, "engine", eng)
        dm.init_db()
        dm.init_db()  # idempotent
        insp = inspect(eng)
        assert {"raw_data", "training_data", "user_votes", "user_analytics",
                "system_metrics"} <= set(insp.get_table_names())
        names = {i["name"] for t in insp.get_table_names() for i in insp.get_indexes(t)}
        assert {"idx_user_votes_unique", "idx_system_metrics_timestamp",
                "idx_training_data_is_answer_duplicate", "idx_raw_data_admin_approved"} <= names
        eng.dispose()

    def test_init_db_survives_legacy_duplicate_votes(self, tmp_path, monkeypatch, caplog, capsys):
        eng = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
        dm.Base.metadata.create_all(bind=eng)
        with eng.begin() as conn:
            conn.execute(text("DROP INDEX idx_user_votes_unique"))
            conn.execute(text("INSERT INTO raw_data (id, telegram_id, question) VALUES (1, 1, 'q')"))
            conn.execute(text("INSERT INTO user_votes (raw_data_id, telegram_user_id, current_vote) VALUES (1, 9, 1)"))
            conn.execute(text("INSERT INTO user_votes (raw_data_id, telegram_user_id, current_vote) VALUES (1, 9, -1)"))
        monkeypatch.setattr(dm, "engine", eng)
        dm.init_db()  # must not raise
        assert "idx_user_votes_unique" in caplog.text
        eng.dispose()

    def test_get_db_yields_and_closes_session(self, db_session):
        gen = dm.get_db()
        session = next(gen)
        assert session.execute(text("SELECT 1")).scalar() == 1
        with pytest.raises(StopIteration):
            next(gen)
        # closed session holds no connection/transaction any more
        assert not session.in_transaction()


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

class TestSimilarity:
    @pytest.mark.parametrize("a,b", [
        ("How can I apply for an internship", "how can i apply for the internship program"),
        ("When is the final exam of algorithms", "When is the midterm exam of algorithms"),
        ("cafeteria menu link", "Where is the cafeteria menu link?"),
        ("text document about exams", "exam document text"),  # dummy-doc terms
        ("course 341 credits", "course 342 credits"),
        ("apple apple banana", "apple banana banana cherry"),
    ])
    def test_matches_previous_sklearn_scoring(self, a, b):
        """Same TF-IDF scoring as before (so thresholds/stored scores keep
        their meaning) for plain-ASCII text."""
        assert dm.calculate_cosine_similarity(a, b) == pytest.approx(sklearn_reference_score(a, b), abs=1e-9)

    def test_identical_and_symmetric(self):
        a, b = "Bölüm başkanı kim?", "Bölüm başkanı kimdir"
        assert dm.calculate_cosine_similarity(a, a) == 1.0
        assert dm.calculate_cosine_similarity(a, b) == dm.calculate_cosine_similarity(b, a)
        assert 0.0 < dm.calculate_cosine_similarity(a, b) < 1.0

    @pytest.mark.parametrize("a,b", [("", "x"), (None, "x"), ("x", None), (None, None), ("   ", "  ")])
    def test_empty_inputs(self, a, b):
        assert dm.calculate_cosine_similarity(a, b) == 0.0

    def test_digits_are_significant(self):
        # digits used to be dropped -> these scored 1.0
        assert dm.calculate_cosine_similarity("Fizik 3 kaç akts", "Fizik 4 kaç akts") < 0.9
        assert dm.calculate_cosine_similarity("BIL 341 dersi ne zaman", "BIL 342 dersi ne zaman") < 0.9
        assert dm.calculate_cosine_similarity("örüntüyü devam ettir: 1, 1, 1, 1",
                                              "örüntüyü devam ettir: 1, 2, 4, 8") < 0.9

    @pytest.mark.parametrize("a,b", [
        ("İSTANBUL ÜNİVERSİTESİ", "istanbul üniversitesi"),
        ("Öğrenci işleri nerede?", "ogrenci isleri nerede"),
        ("IŞIK", "ışık"),
        ("Digital İmage process", "digital image process"),
    ])
    def test_turkish_case_and_diacritics_insensitive(self, a, b):
        assert dm.calculate_cosine_similarity(a, b) == 1.0

    def test_clean_text_turkish_capital_dotted_i(self):
        assert dm.clean_text("İzmir") == "izmir"
        assert "̇" not in dm.clean_text("İSTANBUL")

    @pytest.mark.parametrize("a,b,expected", [
        ("the a an", "the a an", 1.0),       # stop-words only (used to risk "empty vocabulary")
        ("?!", "?!", 1.0),                   # no letters at all
        ("?! .", "?! ,", pytest.approx(1 / 3)),
        ("😃", "😥", 0.0),                   # emoji only -> nothing to compare
        ("a", "b", 0.0),
        ("x", "x y", pytest.approx(sklearn_reference_score("x", "x y"))),
        ("?!", "hello", 0.0),
    ])
    def test_degenerate_texts_never_crash(self, a, b, expected):
        assert dm.calculate_cosine_similarity(a, b) == expected

    def test_no_sklearn_fit_per_comparison(self, db_session, monkeypatch):
        """The hot path must not build/fit an sklearn vectorizer per pair."""
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise AssertionError("TfidfVectorizer must not be used on the hot path")
        monkeypatch.setattr(dm, "TfidfVectorizer", boom)
        for i in range(5):
            add_question(db_session, f"Yaz okulu kaç ders alınabilir {i}")
        q = add_question(db_session, "Yaz okulunda kaç ders alınabilir")
        assert dm.calculate_cosine_similarity("a b", "a c") > 0
        assert dm.mark_duplicate_questions(db_session, q.id, q.question) is True
        assert calls == []

    def test_thread_safe(self):
        texts = [f"soru {i} staj başvurusu {i % 7} ne zaman ders {i % 3}" for i in range(40)]
        pairs = [(a, b) for a in texts for b in texts[:10]]
        expected = [dm.calculate_cosine_similarity(a, b) for a, b in pairs]
        dm._prepare.cache_clear()
        results = [None] * 8
        errors = []

        def worker(k):
            try:
                results[k] = [dm.calculate_cosine_similarity(a, b) for a, b in pairs]
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert all(r == expected for r in results)


# --------------------------------------------------------------------------- #
# find_similar_questions / find_similar_answers
# --------------------------------------------------------------------------- #

class TestFindSimilar:
    def test_find_similar_questions_basic(self, db_session):
        a = add_question(db_session, "Merkezi kafeterya günlük yemek bilgileri hangi linkte?")
        add_question(db_session, "Bölüm başkanı kim?")
        res = dm.find_similar_questions(db_session, "merkezi kafeterya yemek bilgileri hangi linkte")
        assert [r[0] for r in res] == [a.id]
        assert res[0][1] == a.question
        assert 0.45 <= res[0][2] <= 1.0

    def test_sorted_by_score_then_id(self, db_session):
        q1 = add_question(db_session, "staj başvuru tarihleri")
        q2 = add_question(db_session, "staj başvuru tarihleri")
        q3 = add_question(db_session, "staj başvuru tarihleri nelerdir")
        res = dm.find_similar_questions(db_session, "staj başvuru tarihleri")
        assert [r[0] for r in res] == [q1.id, q2.id, q3.id]
        assert res[0][2] == res[1][2] == 1.0 > res[2][2]

    def test_window_is_bounded(self, db_session):
        rows = [RawData(telegram_id=1, question="aynı soru") for _ in range(130)]
        db_session.add_all(rows)
        db_session.commit()
        assert len(dm.find_similar_questions(db_session, "aynı soru")) == 100
        assert len(dm.find_similar_questions(db_session, "aynı soru", limit=10)) == 10
        assert len(dm.find_similar_questions(db_session, "aynı soru", limit=None)) == 130

    def test_exclude_and_before_id(self, db_session):
        ids = [add_question(db_session, "yaz okulu var mı").id for _ in range(5)]
        res = dm.find_similar_questions(db_session, "yaz okulu var mı", exclude_id=ids[2])
        assert ids[2] not in [r[0] for r in res] and len(res) == 4
        res = dm.find_similar_questions(db_session, "yaz okulu var mı", before_id=ids[3])
        assert sorted(r[0] for r in res) == ids[:3]
        res = dm.find_similar_questions(db_session, "yaz okulu var mı", before_id=ids[4], limit=2)
        assert sorted(r[0] for r in res) == ids[2:4]  # the 2 right before it

    def test_empty_query_returns_empty(self, db_session):
        add_question(db_session, "x")
        assert dm.find_similar_questions(db_session, "") == []
        assert dm.find_similar_questions(db_session, None) == []
        assert dm.find_similar_answers(db_session, "") == []

    def test_db_error_returns_empty_and_session_usable(self, db_session):
        db_session.execute(text("DROP TABLE raw_data"))
        db_session.commit()
        assert dm.find_similar_questions(db_session, "soru") == []
        assert db_session.execute(text("SELECT 1")).scalar() == 1

    def test_find_similar_answers(self, db_session):
        a = add_training(db_session, "q1", "Staj başvuruları her yıl Mayıs ayında öğrenci işleri üzerinden yapılır.")
        add_training(db_session, "q2", "Bölüm başkanı Prof. Dr. X'tir.")
        c = add_training(db_session, "q3", "Staj başvuruları her yıl Mayıs ayında öğrenci işleri üzerinden yapılır!")
        res = dm.find_similar_answers(db_session, "staj başvuruları her yıl mayıs ayında öğrenci işleri üzerinden yapılır")
        assert [r[0] for r in res] == [a.id, c.id]
        res = dm.find_similar_answers(db_session, c.answer, before_id=c.id)
        assert [r[0] for r in res] == [a.id]
        res = dm.find_similar_answers(db_session, c.answer, exclude_id=a.id)
        assert [r[0] for r in res] == [c.id]


# --------------------------------------------------------------------------- #
# mark_duplicate_questions / mark_duplicate_answers
# --------------------------------------------------------------------------- #

CAFE_1 = "Merkezi kafeterya günlük yemek bilgileri hangi linkte bulunuyor?"
CAFE_2 = "Merkezi kafeterya günlük yemek bilgileri hangi linkte?"
CAFE_3 = "merkezi kafeterya yemek bilgileri hangi linkte bulunuyor"


class TestMarkDuplicateQuestions:
    def test_new_question_marked_against_oldest(self, db_session):
        a = add_question(db_session, CAFE_1)
        add_question(db_session, "Bölüm başkanı kim?")
        b = add_question(db_session, CAFE_2)
        c = add_question(db_session, CAFE_3)
        assert dm.mark_duplicate_questions(db_session, c.id, c.question) is True
        c = reload(db_session, RawData, c.id)
        assert c.is_duplicate is True
        assert c.duplicate_of_id == a.id
        assert c.similarity_score == pytest.approx(dm.calculate_cosine_similarity(CAFE_3, CAFE_1))
        # older rows are left alone
        assert reload(db_session, RawData, a.id).is_duplicate is False
        assert reload(db_session, RawData, b.id).is_duplicate is False

    def test_no_match_returns_false(self, db_session):
        add_question(db_session, "Bölüm başkanı kim?")
        q = add_question(db_session, "Yaz okulu ücretleri ne kadar")
        assert dm.mark_duplicate_questions(db_session, q.id, q.question) is False
        assert reload(db_session, RawData, q.id).is_duplicate is False

    def test_first_question_never_marked(self, db_session):
        """Re-checking the oldest question (e.g. on admin approval) used to mark
        it as a duplicate of a NEWER question and re-point other rows."""
        a = add_question(db_session, CAFE_1)
        b = add_question(db_session, CAFE_2)
        assert dm.mark_duplicate_questions(db_session, a.id, a.question) is False
        assert reload(db_session, RawData, a.id).is_duplicate is False
        assert reload(db_session, RawData, b.id).duplicate_of_id is None

    @pytest.mark.parametrize("order", ["asc", "desc"])
    def test_full_recheck_has_no_cycles(self, db_session, order):
        """Admin /detect-duplicates: clear flags, then check every row. The old
        code ended with 1->5, 5->1 style cycles and no original left."""
        a = add_question(db_session, CAFE_1)
        add_question(db_session, "Bölüm başkanı kim?")
        b = add_question(db_session, CAFE_2)
        c = add_question(db_session, CAFE_3)
        entries = db_session.query(RawData).order_by(
            RawData.id.asc() if order == "asc" else RawData.id.desc()).all()
        for e in entries:
            dm.mark_duplicate_questions(db_session, e.id, e.question)
        db_session.expire_all()
        assert reload(db_session, RawData, a.id).is_duplicate is False
        assert reload(db_session, RawData, b.id).duplicate_of_id == a.id
        assert reload(db_session, RawData, c.id).duplicate_of_id == a.id
        for row in db_session.query(RawData).filter(RawData.is_duplicate == True):  # noqa: E712
            assert row.duplicate_of_id < row.id

    def test_points_to_root_of_existing_group(self, db_session):
        root = add_question(db_session, "Yemekhane menüsü nereden görülür")
        dup = add_question(db_session, CAFE_1, is_duplicate=True, duplicate_of_id=root.id,
                           similarity_score=0.5)
        new = add_question(db_session, CAFE_2)
        assert dm.calculate_cosine_similarity(new.question, root.question) < 0.45
        assert dm.mark_duplicate_questions(db_session, new.id, new.question) is True
        new = reload(db_session, RawData, new.id)
        assert new.duplicate_of_id == root.id
        assert new.similarity_score == pytest.approx(dm.calculate_cosine_similarity(CAFE_2, CAFE_1))
        assert reload(db_session, RawData, dup.id).duplicate_of_id == root.id

    def test_legacy_cycle_and_dangling_links_terminate(self, db_session):
        a = add_question(db_session, "Yemekhane menüsü nereden görülür")
        b = add_question(db_session, CAFE_1, is_duplicate=True)
        a.is_duplicate, a.duplicate_of_id = True, b.id
        b.duplicate_of_id = a.id
        db_session.commit()
        new = add_question(db_session, CAFE_2)
        assert dm.mark_duplicate_questions(db_session, new.id, new.question) is True
        assert reload(db_session, RawData, new.id).duplicate_of_id in (a.id, b.id)

        d = add_question(db_session, "Staj defteri teslim tarihi", is_duplicate=True,
                         duplicate_of_id=999999)
        e = add_question(db_session, "staj defteri teslim tarihi nedir")
        assert dm.mark_duplicate_questions(db_session, e.id, e.question) is True
        assert reload(db_session, RawData, e.id).duplicate_of_id == d.id

    def test_error_rolls_back_and_returns_false(self, db_session, monkeypatch):
        add_question(db_session, CAFE_1)
        q = add_question(db_session, CAFE_2)
        real_commit = db_session.commit

        def failing_commit():
            raise OperationalError("COMMIT", {}, Exception("database is locked"))

        monkeypatch.setattr(db_session, "commit", failing_commit)
        assert dm.mark_duplicate_questions(db_session, q.id, q.question) is False
        monkeypatch.setattr(db_session, "commit", real_commit)
        # pending change was rolled back, session still usable
        assert reload(db_session, RawData, q.id).is_duplicate is False
        assert db_session.execute(text("SELECT 1")).scalar() == 1


class TestMarkDuplicateAnswers:
    ANSWER = "Staj başvuruları her yıl Mayıs ayında öğrenci işleri üzerinden yapılır."

    def test_marks_most_similar_older_answer(self, db_session):
        a = add_training(db_session, "q1", self.ANSWER)
        add_training(db_session, "q2", "Bölüm başkanı Prof. Dr. X'tir.")
        b = add_training(db_session, "q3", self.ANSWER + " Detaylar için web sitesine bakın.")
        c = add_training(db_session, "q4", self.ANSWER)
        assert dm.mark_duplicate_answers(db_session, c.id, c.answer) is True
        c = reload(db_session, TrainingData, c.id)
        assert c.is_answer_duplicate is True
        assert c.duplicate_answer_of_id == a.id  # 1.0 beats b's score
        assert c.answer_similarity_score == 1.0
        assert reload(db_session, TrainingData, b.id).is_answer_duplicate is False

    def test_below_threshold_not_marked(self, db_session):
        add_training(db_session, "q1", self.ANSWER)
        b = add_training(db_session, "q2", "Staj başvuruları Eylül ayında bölüm sekreterliğine yapılır.")
        assert dm.mark_duplicate_answers(db_session, b.id, b.answer) is False

    def test_recheck_both_directions_no_mutual_duplicates(self, db_session):
        a = add_training(db_session, "q1", self.ANSWER)
        b = add_training(db_session, "q2", self.ANSWER + "!")
        for row in (b, a, a, b):
            dm.mark_duplicate_answers(db_session, row.id, row.answer)
        assert reload(db_session, TrainingData, a.id).is_answer_duplicate is False
        assert reload(db_session, TrainingData, b.id).duplicate_answer_of_id == a.id

    def test_missing_entry_returns_false(self, db_session):
        add_training(db_session, "q1", self.ANSWER)
        assert dm.mark_duplicate_answers(db_session, 12345, self.ANSWER) is False


# --------------------------------------------------------------------------- #
# process_new_question / process_new_training_data
# --------------------------------------------------------------------------- #

class TestProcessNew:
    def test_process_new_question(self, db_session):
        a = dm.process_new_question(db_session, {"telegram_id": 1, "question": CAFE_1, "language": "TR"})
        b = dm.process_new_question(db_session, {"telegram_id": 2, "question": CAFE_2, "language": "TR"})
        assert a.id and b.id
        assert a.is_duplicate is False
        assert b.is_duplicate is True and b.duplicate_of_id == a.id

    def test_invalid_payload_raises_and_rolls_back(self, db_session):
        with pytest.raises(TypeError):
            dm.process_new_question(db_session, {"telegram_id": 1, "question": "q", "nope": 1})
        with pytest.raises(IntegrityError):
            dm.process_new_question(db_session, {"telegram_id": 1})  # question NOT NULL
        assert db_session.query(RawData).count() == 0  # session usable after rollback

    def test_duplicate_detection_failure_keeps_question(self, db_session, monkeypatch):
        def broken(*a, **k):
            raise RuntimeError("similarity backend down")
        monkeypatch.setattr(dm, "find_similar_questions", broken)
        row = dm.process_new_question(db_session, {"telegram_id": 1, "question": "q"})
        assert reload(db_session, RawData, row.id) is not None

    def test_process_new_training_data(self, db_session):
        a = dm.process_new_training_data(db_session, {"question": "q", "answer": TestMarkDuplicateAnswers.ANSWER})
        b = dm.process_new_training_data(db_session, {"question": "q2", "answer": TestMarkDuplicateAnswers.ANSWER})
        assert a.is_answer_duplicate is False
        assert b.is_answer_duplicate is True and b.duplicate_answer_of_id == a.id
        with pytest.raises(IntegrityError):
            dm.process_new_training_data(db_session, {"question": "q3"})  # answer NOT NULL
        assert db_session.query(TrainingData).count() == 2


# --------------------------------------------------------------------------- #
# Voting
# --------------------------------------------------------------------------- #

def vote_row(db, raw_id, user):
    db.expire_all()
    return db.query(UserVotes).filter_by(raw_data_id=raw_id, telegram_user_id=user).one()


class TestHandleUserVote:
    def test_first_vote(self, db_session):
        q = add_question(db_session, "q")
        r = dm.handle_user_vote(db_session, q.id, 100, 1)
        assert r == {"success": True, "message": "👍 Liked! (2 changes available)", "changes_left": 2}
        v = vote_row(db_session, q.id, 100)
        assert v.current_vote == 1 and v.vote_changes_count == 0
        assert v.first_vote_at is not None and v.last_vote_at is not None

    def test_same_vote_twice(self, db_session):
        q = add_question(db_session, "q")
        dm.handle_user_vote(db_session, q.id, 100, -1)
        r = dm.handle_user_vote(db_session, q.id, 100, -1)
        assert r == {"success": False, "message": "You already voted this way!", "changes_left": 2}
        assert vote_row(db_session, q.id, 100).vote_changes_count == 0
        assert db_session.query(UserVotes).count() == 1

    def test_change_limit(self, db_session):
        q = add_question(db_session, "q")
        assert dm.handle_user_vote(db_session, q.id, 100, 1)["changes_left"] == 2
        r1 = dm.handle_user_vote(db_session, q.id, 100, -1)
        assert r1 == {"success": True, "message": "👎 Disliked! (1 changes left)", "changes_left": 1}
        r2 = dm.handle_user_vote(db_session, q.id, 100, 1)
        assert r2 == {"success": True, "message": "👍 Liked! (Last change!)", "changes_left": 0}
        r3 = dm.handle_user_vote(db_session, q.id, 100, -1)
        assert r3 == {"success": False, "message": "No vote changes left! (2/2 used)", "changes_left": 0}
        v = vote_row(db_session, q.id, 100)
        assert v.current_vote == 1 and v.vote_changes_count == 2
        # clicking the vote you already have is reported as such
        r4 = dm.handle_user_vote(db_session, q.id, 100, 1)
        assert r4 == {"success": False, "message": "You already voted this way!", "changes_left": 0}

    def test_last_vote_at_updated_on_change(self, db_session):
        q = add_question(db_session, "q")
        dm.handle_user_vote(db_session, q.id, 100, 1)
        db_session.execute(text("UPDATE user_votes SET last_vote_at = '2000-01-01 00:00:00'"))
        db_session.commit()
        dm.handle_user_vote(db_session, q.id, 100, -1)
        v = vote_row(db_session, q.id, 100)
        assert v.last_vote_at > datetime(2000, 1, 2)

    @pytest.mark.parametrize("bad", [0, 2, -2, None, "1"])
    def test_invalid_vote_type(self, db_session, bad):
        q = add_question(db_session, "q")
        r = dm.handle_user_vote(db_session, q.id, 100, bad)
        assert r["success"] is False
        assert db_session.query(UserVotes).count() == 0

    def test_unknown_question(self, db_session):
        r = dm.handle_user_vote(db_session, 4242, 100, 1)
        assert r == {"success": False, "message": "Record not found!", "changes_left": 0}
        assert db_session.query(UserVotes).count() == 0

    def test_legacy_null_change_count(self, db_session):
        q = add_question(db_session, "q")
        db_session.add(UserVotes(raw_data_id=q.id, telegram_user_id=5, current_vote=1))
        db_session.commit()
        db_session.execute(text("UPDATE user_votes SET vote_changes_count = NULL"))
        db_session.commit()
        r = dm.handle_user_vote(db_session, q.id, 5, -1)
        assert r["success"] is True and r["changes_left"] == 1
        assert vote_row(db_session, q.id, 5).vote_changes_count == 1

    def test_raw_data_like_follows_votes(self, db_session):
        """raw_data.like (admin panel 'point' + liked/disliked stats) used to
        stay NULL forever because only user_votes was written."""
        q = add_question(db_session, "q")
        other = add_question(db_session, "other")
        dm.handle_user_vote(db_session, q.id, 1, 1)
        assert reload(db_session, RawData, q.id).like == 1
        dm.handle_user_vote(db_session, q.id, 2, -1)
        assert reload(db_session, RawData, q.id).like is None  # tie
        dm.handle_user_vote(db_session, q.id, 3, -1)
        assert reload(db_session, RawData, q.id).like == -1
        dm.handle_user_vote(db_session, q.id, 3, 1)  # change back
        dm.handle_user_vote(db_session, q.id, 4, 1)
        assert reload(db_session, RawData, q.id).like == 1
        assert reload(db_session, RawData, other.id).like is None
        stats = dm.get_database_statistics(db_session)
        assert stats["liked_questions"] == 1 and stats["disliked_questions"] == 0

    def test_db_error_returns_error_and_session_usable(self, db_session):
        q = add_question(db_session, "q")
        db_session.execute(text("DROP TABLE user_votes"))
        db_session.commit()
        r = dm.handle_user_vote(db_session, q.id, 1, 1)
        assert r == {"success": False, "message": "Error saving vote", "changes_left": 0}
        assert reload(db_session, RawData, q.id).question == "q"

    def test_concurrent_insert_race_is_handled(self, db_session, other_session, monkeypatch):
        """Double click: another request inserts the vote between our SELECT
        and INSERT. The UNIQUE index rejects ours; we must not report an
        error nor leave the session broken."""
        q = add_question(db_session, "q")
        real_flush = db_session.flush
        state = {"done": False}

        def racing_flush(*a, **k):
            if not state["done"]:
                state["done"] = True
                s2 = other_session()
                s2.add(UserVotes(raw_data_id=q.id, telegram_user_id=77, current_vote=1,
                                 vote_changes_count=0))
                s2.commit()
            return real_flush(*a, **k)

        monkeypatch.setattr(db_session, "flush", racing_flush)
        r = dm.handle_user_vote(db_session, q.id, 77, 1)
        assert state["done"], "race was not simulated"
        assert r == {"success": False, "message": "You already voted this way!", "changes_left": 2}
        assert db_session.query(UserVotes).count() == 1

    def test_concurrent_change_race_cannot_exceed_limit(self, db_session, other_session, monkeypatch):
        """Two change requests read the same vote row; only one may use the
        change (no lost update of vote_changes_count)."""
        q = add_question(db_session, "q")
        dm.handle_user_vote(db_session, q.id, 77, 1)
        real_execute = db_session.execute
        state = {"done": False}

        def racing_execute(stmt, *a, **k):
            if not state["done"] and getattr(stmt, "is_update", False) \
                    and getattr(getattr(stmt, "table", None), "name", None) == "user_votes":
                state["done"] = True
                s2 = other_session()
                assert dm.handle_user_vote(s2, q.id, 77, -1)["success"] is True
            return real_execute(stmt, *a, **k)

        monkeypatch.setattr(db_session, "execute", racing_execute)
        r = dm.handle_user_vote(db_session, q.id, 77, -1)
        assert state["done"], "race was not simulated"
        assert r == {"success": False, "message": "You already voted this way!", "changes_left": 1}
        v = vote_row(db_session, q.id, 77)
        assert v.current_vote == -1 and v.vote_changes_count == 1

    def test_threaded_double_clicks_keep_counters_consistent(self, db_session):
        q = add_question(db_session, "q")
        qid = q.id
        n_threads = 8
        barrier = threading.Barrier(n_threads)
        results, errors = [], []
        lock = threading.Lock()

        def click(k):
            s = dm.SessionLocal()
            try:
                barrier.wait()
                for i in range(4):
                    r = dm.handle_user_vote(s, qid, 555, 1 if (k + i) % 2 else -1)
                    with lock:
                        results.append(r)
            except Exception as e:  # pragma: no cover
                errors.append(e)
            finally:
                s.close()

        threads = [threading.Thread(target=click, args=(k,)) for k in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not errors
        rows = db_session.query(UserVotes).filter_by(raw_data_id=qid).all()
        assert len(rows) == 1
        successes = sum(1 for r in results if r["success"])
        # 1 creation + at most 2 changes, and every success is accounted for
        assert rows[0].vote_changes_count <= 2
        assert successes == 1 + rows[0].vote_changes_count
        expected_like = rows[0].current_vote
        assert reload(db_session, RawData, qid).like == expected_like


class TestSyncLikeFromVotes:
    def test_backfill_all(self, db_session):
        q1 = add_question(db_session, "q1")
        q2 = add_question(db_session, "q2")
        q3 = add_question(db_session, "q3", like=1)  # legacy value, no votes
        q4 = add_question(db_session, "q4")
        db_session.add_all([
            UserVotes(raw_data_id=q1.id, telegram_user_id=1, current_vote=1),
            UserVotes(raw_data_id=q1.id, telegram_user_id=2, current_vote=1),
            UserVotes(raw_data_id=q2.id, telegram_user_id=1, current_vote=-1),
            UserVotes(raw_data_id=q4.id, telegram_user_id=1, current_vote=1),
            UserVotes(raw_data_id=q4.id, telegram_user_id=2, current_vote=-1),
        ])
        db_session.commit()
        assert dm.sync_like_from_votes(db_session) == 3
        assert reload(db_session, RawData, q1.id).like == 1
        assert reload(db_session, RawData, q2.id).like == -1
        assert reload(db_session, RawData, q3.id).like == 1  # untouched
        assert reload(db_session, RawData, q4.id).like is None

    def test_single_row(self, db_session):
        q1 = add_question(db_session, "q1")
        q2 = add_question(db_session, "q2")
        db_session.add_all([
            UserVotes(raw_data_id=q1.id, telegram_user_id=1, current_vote=-1),
            UserVotes(raw_data_id=q2.id, telegram_user_id=1, current_vote=-1),
        ])
        db_session.commit()
        assert dm.sync_like_from_votes(db_session, q1.id) == 1
        assert reload(db_session, RawData, q1.id).like == -1
        assert reload(db_session, RawData, q2.id).like is None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

class TestStatistics:
    def test_vote_statistics(self, db_session):
        q = add_question(db_session, "q")
        other = add_question(db_session, "other")
        db_session.add_all([
            UserVotes(raw_data_id=q.id, telegram_user_id=1, current_vote=1),
            UserVotes(raw_data_id=q.id, telegram_user_id=2, current_vote=1),
            UserVotes(raw_data_id=q.id, telegram_user_id=3, current_vote=-1),
            UserVotes(raw_data_id=q.id, telegram_user_id=4, current_vote=None),
            UserVotes(raw_data_id=other.id, telegram_user_id=1, current_vote=-1),
        ])
        db_session.commit()
        assert dm.get_vote_statistics(db_session, q.id) == {
            "likes": 2, "dislikes": 1, "total_votes": 4, "score": 1}
        assert dm.get_vote_statistics(db_session, 999) == {
            "likes": 0, "dislikes": 0, "total_votes": 0, "score": 0}
        stats = dm.get_vote_statistics(db_session, q.id)
        assert all(type(v) is int for v in stats.values())

    def test_vote_statistics_error_path(self, db_session):
        db_session.execute(text("DROP TABLE user_votes"))
        db_session.commit()
        assert dm.get_vote_statistics(db_session, 1) == {
            "likes": 0, "dislikes": 0, "total_votes": 0, "score": 0}
        assert db_session.execute(text("SELECT 1")).scalar() == 1

    def test_database_statistics(self, db_session):
        add_question(db_session, "a", answer="x", language="TR", like=1, admin_approved=1,
                     quality_score=0.5, processing_time=2.0)
        add_question(db_session, "b", language="TR", like=-1, is_duplicate=True,
                     quality_score=1.0, processing_time=4.0)
        add_question(db_session, "c", language="EN")
        add_question(db_session, "d")
        add_training(db_session, "a", "x")
        add_training(db_session, "b", "y", is_active=False, is_answer_duplicate=True)
        s = dm.get_database_statistics(db_session)
        assert s["total_questions"] == 4
        assert s["answered_questions"] == 1
        assert s["liked_questions"] == 1
        assert s["disliked_questions"] == 1
        assert s["approved_questions"] == 1
        assert s["duplicate_questions"] == 1
        assert s["training_data_count"] == 2
        assert s["active_training_data"] == 1
        assert s["duplicate_answers"] == 1
        assert s["language_distribution"] == {"TR": 2, "EN": 1, None: 1}
        assert s["avg_quality_score"] == pytest.approx(0.75)
        assert s["avg_response_time"] == pytest.approx(3.0)

    def test_database_statistics_empty_and_error(self, db_session):
        s = dm.get_database_statistics(db_session)
        assert s["total_questions"] == 0 and s["avg_quality_score"] == 0.0
        db_session.execute(text("DROP TABLE training_data"))
        db_session.commit()
        assert dm.get_database_statistics(db_session) == {}
        assert db_session.execute(text("SELECT 1")).scalar() == 1
