import mongomock
import pytest
from pydantic import ValidationError

from src.core.models import FinancialGoal, Holding, Portfolio, UserProfile
from src.data.user_store import AccountError, NewAccount, UserStore


@pytest.fixture
def store() -> UserStore:
    store = UserStore(mongomock.MongoClient()["finnie_test"])
    store.ensure_indexes()
    return store


@pytest.fixture
def profile() -> UserProfile:
    goal = FinancialGoal(name="House", target_amount=60000, horizon_years=5, risk_tolerance="moderate")
    return UserProfile(session_id="s1", knowledge_level="intermediate", risk_tolerance="moderate",
                       goals=[goal], interests=["ETFs"])


def new_account(email: str = "Ada@Example.com") -> NewAccount:
    return NewAccount(name="Ada", email=email, password="correct horse")


def test_create_account_stores_user_profile_and_portfolio(store, profile):
    portfolio = Portfolio(holdings=[Holding(ticker="vti", quantity=10, cost_basis=220)])
    account = store.create_account(new_account(), profile, portfolio)

    assert account.email == "ada@example.com"
    user = store.users.find_one({"email": "ada@example.com"})
    assert user["password_hash"] != "correct horse"
    assert "correct horse" not in str(user)

    loaded = store.load_profile(account.id, session_id="s2")
    assert loaded.session_id == "s2"
    assert loaded.knowledge_level == "intermediate"
    assert loaded.goals[0].name == "House"
    assert store.load_portfolio(account.id).holdings[0].ticker == "VTI"
    assert store.load_portfolio(account.id).market == "US"


def test_portfolio_is_optional(store, profile):
    account = store.create_account(new_account(), profile)
    assert store.load_portfolio(account.id) is None


def test_duplicate_email_is_rejected(store, profile):
    store.create_account(new_account(), profile)
    with pytest.raises(AccountError):
        store.create_account(new_account("ada@example.com "), profile)


def test_failed_profile_save_removes_the_user(store, profile, mocker):
    mocker.patch.object(store, "save_profile", side_effect=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        store.create_account(new_account(), profile)
    assert store.users.count_documents({}) == 0


def test_authenticate(store, profile):
    account = store.create_account(new_account(), profile)
    assert store.authenticate(" ADA@example.com", "correct horse") == account
    assert store.authenticate("ada@example.com", "wrong password") is None
    assert store.authenticate("nobody@example.com", "correct horse") is None


def test_save_profile_updates_level_and_risk(store, profile):
    account = store.create_account(new_account(), profile)
    store.save_profile(account.id, profile.model_copy(update={"knowledge_level": "advanced"}))
    assert store.load_profile(account.id, "s1").knowledge_level == "advanced"
    assert store.profiles.count_documents({}) == 1


@pytest.mark.parametrize("fields", [
    {"name": " ", "email": "a@b.co", "password": "longenough"},
    {"name": "Ada", "email": "not-an-email", "password": "longenough"},
    {"name": "Ada", "email": "a@b.co", "password": "short"},
    {"name": "Ada", "email": "a@b.co", "password": "x" * 73},
])
def test_new_account_validation(fields):
    with pytest.raises(ValidationError):
        NewAccount(**fields)


def test_update_name_and_delete_portfolio(store, profile):
    portfolio = Portfolio(holdings=[Holding(ticker="VTI", quantity=1)])
    account = store.create_account(new_account(), profile, portfolio)
    store.update_name(account.id, "Ada L.")
    assert store.authenticate("ada@example.com", "correct horse").name == "Ada L."
    store.delete_portfolio(account.id)
    assert store.load_portfolio(account.id) is None
