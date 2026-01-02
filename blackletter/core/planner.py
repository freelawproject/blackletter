"""Phase 2: Planning redactions using state machine."""

import logging
from enum import Enum
from typing import Dict, List

from blackletter import Document
from blackletter.config import RedactionConfig
from blackletter.core.scanner import Detection, Opinion

logger = logging.getLogger(__name__)


class OpinionState(Enum):
    """State machine for opinion detection."""

    WAIT_CAPTION = "WAIT_CAPTION"
    TRACKING = "TRACKING"
    LOCKED_UNTIL_KEY = "LOCKED_UNTIL_KEY"


class OpinionPlanner:
    """Plans redaction instructions using a state machine."""

    def __init__(self, config: RedactionConfig):
        self.config = config
        self.opinion_idx = 0

    def plan(
        self,
        document: Document,
    ) -> Document:
        """Plan redaction instructions and identify opinion spans.

        Returns:
            - redaction_instructions: List of start->end redaction pairs
            - opinion_spans: List of detected opinions with metadata
            - page_headers: Dict mapping page_idx to header y-coordinate
            - page_footers: Dict mapping page_idx to footer y-coordinate
        """
        logger.info("Starting PHASE 2: Planning redactions")

        # State machine variables
        candidate_end_node = None
        current_state = OpinionState.WAIT_CAPTION
        document.sort_all_objects()

        current_opinion = None
        # State machine - process objects in order
        for page in document.pages:
            for obj in page.page_objects:
                label = obj.label
                if label not in ["caption", "line", "headmatter", "Key"]:
                    continue

                # LOCKED: waiting for Key to end the opinion
                if current_state == OpinionState.LOCKED_UNTIL_KEY:
                    if label == "Key":
                        current_opinion.key = obj
                        document.opinions.append(current_opinion)
                        current_opinion = None
                        current_state = OpinionState.WAIT_CAPTION
                    continue

                # WAITING: looking for a caption to start
                if current_state == OpinionState.WAIT_CAPTION:
                    if label == "caption":
                        current_opinion = Opinion(caption=obj)
                        current_state = OpinionState.TRACKING
                    continue

                # TRACKING: looking for line/headmatter or next caption/Key
                if current_state == OpinionState.TRACKING:
                    if label == "line":
                        current_state = OpinionState.LOCKED_UNTIL_KEY
                        current_opinion.line = obj

                    elif label == "headmatter":
                        if current_opinion.headmatter == None:
                            current_opinion.headmatter = obj

                        if candidate_end_node is None:
                            candidate_end_node = obj

                    elif label == "Key":
                        current_opinion.key = obj
                        document.opinions.append(current_opinion)
                        current_opinion = None
                        current_state = OpinionState.WAIT_CAPTION
                        candidate_end_node = None

        document.assign_case_names()

        logger.info(f"Planned {len(document.opinions)} opinions")
        return document

    def _record_opinion_span(self, spans: List, start: Detection, end: Detection, reason: str):
        """Record an opinion span."""
        if not start or not end:
            return

        self.opinion_idx += 1
        spans.append({"n": self.opinion_idx, "start": start, "end": end, "reason": reason})

        sp = start.page_index + 1
        ep = end.page_index + 1
        logger.info(f"Opinion {self.opinion_idx:03d}: pages {sp}–{ep} ({reason})")

    @staticmethod
    def _assign_case_names(opinion_spans: List[Dict], page_start: int):
        """Assign case names to opinions."""
        if not opinion_spans:
            return

        COL_ORDER = {"LEFT": 0, "RIGHT": 1}

        def sort_key(span):
            start = span.get("start", {})
            page = start.page_index
            col = COL_ORDER.get(start.col)
            y1 = start.coords[1]
            return (page, col, y1)

        opinion_spans.sort(key=sort_key)

        page_counter = {}
        for sp in opinion_spans:
            first_page = sp["start"].page_index + page_start
            counter = page_counter.get(first_page, 0) + 1
            page_counter[first_page] = counter
            sp["case_name"] = f"{first_page:04d}-{counter:02d}"
