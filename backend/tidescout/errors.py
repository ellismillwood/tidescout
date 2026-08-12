class SourceUnavailable(Exception):
    """An external data source failed and no cached fallback exists."""

    def __init__(self, source: str, detail: str = ""):
        self.source = source
        self.detail = detail
        super().__init__(f"{source} unavailable{': ' + detail if detail else ''}")
