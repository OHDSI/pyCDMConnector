# Copyright 2025 DARWIN EU
# SPDX-License-Identifier: Apache-2.0

"""CDMConnector settings container."""


class Settings:
    """Simple dict-like settings container."""

    def __init__(self, data: dict | None = None):
        self.data: dict = dict(data) if data else {}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __repr__(self):
        return f"Settings({self.data!r})"
