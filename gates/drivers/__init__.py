"""Drivers run traces against an arm through the public surface only —
a gate cannot reach an engine internal that a client could not
(specs/014 design S3.3: the zero-instrumentation deployability trade).
"""

from gates.drivers.sdk import SDKDriver  # noqa: F401
