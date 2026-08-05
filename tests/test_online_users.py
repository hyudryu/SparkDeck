import unittest

from manager import Manager


class OnlineUsersTests(unittest.TestCase):
    def test_unique_users_are_counted_once_across_sessions(self) -> None:
        result = Manager._parse_online_users(
            "mark pts/0 2026-07-28 06:10 (10.0.0.4)\n"
            "hyudryu seat0 2026-07-28 05:00\n"
            "mark pts/1 2026-07-28 06:12 (10.0.0.4)\n"
        )

        self.assertEqual(result, {
            "count": 2,
            "names": ["mark", "hyudryu"],
            "sessions": 3,
        })

    def test_no_sessions_returns_zero_users(self) -> None:
        self.assertEqual(
            Manager._parse_online_users(""),
            {"count": 0, "names": [], "sessions": 0},
        )


if __name__ == "__main__":
    unittest.main()
