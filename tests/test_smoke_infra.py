import database_models


def test_db_redirected_to_temp(db_session):
    url = str(database_models.SessionLocal.kw["bind"].url)
    assert "cengbot_tests_" in url
    assert db_session.query(database_models.RawData).count() == 0
