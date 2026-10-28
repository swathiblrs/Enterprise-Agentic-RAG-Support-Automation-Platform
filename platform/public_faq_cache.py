"""Bounded, expiring cache restricted to curated, public university FAQ text."""
import copy
import json
import threading
import time
from collections import OrderedDict


class PublicFAQCache:
    def __init__(self, cases_path, ttl=3600, max_entries=128, clock=time.monotonic):
        self.allowed = {row["question"] for row in json.loads(cases_path.read_text())}
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def get(self, question, compute):
        if question not in self.allowed:
            result = compute()
            return dict(result, delivery={"public_faq_cache_hit": False, "cache_eligible": False})
        # Single-flight prevents many identical cold requests loading the worker.
        with self.lock:
            now = self.clock()
            cached = self.entries.get(question)
            if cached and now - cached[0] < self.ttl:
                self.entries.move_to_end(question)
                result = copy.deepcopy(cached[1])
                result["delivery"] = {"public_faq_cache_hit": True, "cache_eligible": True}
                return result
            result = compute()
            self.entries[question] = (self.clock(), copy.deepcopy(result))
            self.entries.move_to_end(question)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
            result["delivery"] = {"public_faq_cache_hit": False, "cache_eligible": True}
            return result

    def clear(self):
        with self.lock:
            self.entries.clear()
