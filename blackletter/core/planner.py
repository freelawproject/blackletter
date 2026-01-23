"""Phase 2: Planning redactions using state machine."""

import logging
from enum import Enum

from blackletter.config import RedactionConfig
from blackletter.core.scanner import Document, Opinion

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
                        if current_opinion.headmatter is None:
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
