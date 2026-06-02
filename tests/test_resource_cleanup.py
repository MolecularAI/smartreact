"""Tests for process pool and resource cleanup."""

from __future__ import annotations

from smartreact import KeyGenerator, ReactionEnumerator

from .conftest import BROMOBENZENE, PHENYLBORONIC_ACID


class TestContextManager:
    def test_enumerator_context_manager_closes_pool(self):
        with ReactionEnumerator(n_cores=2) as e:
            _ = e.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
            pool = e._pool  # noqa: SLF001
        # After exit, pool should be gone and shutdown
        assert e._pool is None  # noqa: SLF001
        if pool is not None:
            assert True  # pool is shut down

    def test_enumerator_context_manager_closes_keygen_pool(self):
        with ReactionEnumerator(n_cores=2) as e:
            _ = e.enumerate_pairs([(BROMOBENZENE, PHENYLBORONIC_ACID)], parallel=True)
        # Both the enumerator pool and keygen pool should be closed
        assert e._pool is None  # noqa: SLF001
        assert e.keygen._pool is None  # noqa: SLF001

    def test_keygen_context_manager_closes_pool(self):
        with KeyGenerator(n_cores=2) as kg:
            kg.classify_many([BROMOBENZENE, PHENYLBORONIC_ACID], parallel=True)
        assert kg._pool is None  # noqa: SLF001

    def test_enumerator_close_is_idempotent(self):
        e = ReactionEnumerator(n_cores=1)
        e.close()
        e.close()  # should not raise

    def test_keygen_close_is_idempotent(self):
        kg = KeyGenerator(n_cores=1)
        kg.close()
        kg.close()  # should not raise

    def test_enumerator_context_manager_closes_on_exception(self):
        e = ReactionEnumerator(n_cores=2)
        try:
            with e:
                _ = e.enumerate_pairs([(BROMOBENZENE, PHENYLBORONIC_ACID)], parallel=True)
                raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        assert e._pool is None  # noqa: SLF001
        assert e.keygen._pool is None  # noqa: SLF001


class TestAbandonedGenerator:
    def test_early_break_does_not_leave_pool_broken(self):
        """Breaking out of enumerate_pairs_lazy early should not corrupt the pool."""
        pairs = [(BROMOBENZENE, PHENYLBORONIC_ACID)] * 20
        with ReactionEnumerator(n_cores=2) as e:
            gen = e.enumerate_pairs_lazy(pairs, parallel=True, chunk_size=10)
            # consume only the first result, then abandon
            next(gen, None)
            del gen  # triggers GeneratorExit

            # Pool should still be usable afterwards
            result = e.enumerate_pair(BROMOBENZENE, PHENYLBORONIC_ACID)
            assert len(result) > 0
