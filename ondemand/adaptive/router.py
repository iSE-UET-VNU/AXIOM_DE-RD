import os
import re

from ..bench import BENCH, load_jsonl, work
from ..text import mostly_image
from . import fallback
from .config import get_config
from .registry import register

BRANCHES = ("light", "enrich", "visual")

VISUAL_REGEX = re.compile(
    r"\b(diagram|diagrams|figure|figures|fig|chart|charts|plot|plots|graph|graphs|image|images|picture|table|tables|curve|curves|schematic|schéma|schémas|tableau|tableaux|graphe|graphes|courbe|courbes|dessin|illustration|carte|plan)\b",
    re.IGNORECASE
)


class Router:
    name = "base"

    def route(self, subdata):
        raise NotImplementedError

    def describe(self):
        return self.name


@register("router", "fixed")
class FixedRouter(Router):
    name = "fixed"

    def __init__(self, argument, **_):
        self.branch = argument or "visual"
        if self.branch not in BRANCHES:
            raise SystemExit(f"unknown branch {self.branch!r}; available: {', '.join(BRANCHES)}")

    def route(self, subdata):
        return self.branch, {"reason": "fixed"}

    def describe(self):
        return f"fixed:{self.branch}"


@register("router", "rule")
class RuleRouter(Router):
    name = "rule"

    def __init__(self, argument=None, bench=BENCH, pages_name="pages_ppocrv5.jsonl", chandra_endpoint=None, **_):
        self.argument = argument or "adaptive"
        source = work("light_prep", bench=bench) / pages_name
        self.pages = {r["page_id"]: r for r in load_jsonl(source)} if source.exists() else {}
        if not self.pages:
            fallback.warn("router_pages", f"{source.name} is missing, so the router cannot see which pages were OCR'd "
                                          f"and decides from the query wording alone")
        self.has_chandra = bool(chandra_endpoint or get_config().chandra_endpoint)

    def route(self, subdata):
        candidates = subdata.top(5)
        query = subdata.query

        # Signal 1: Visual intent in query
        has_vis_term = bool(VISUAL_REGEX.search(query))

        # Signal 2: Scanned / OCR / visual-heavy candidates
        ocr_candidates = sum(bool(self.pages.get(p, {}).get("ocr_applied")) for p in candidates)
        visual_only_candidates = sum(bool(self.pages.get(p, {}).get("visual_only")) for p in candidates)
        mostly_img_candidates = sum(mostly_image(self.pages.get(p, {}).get("text", "")) for p in candidates)

        # Signal 3: Score gap
        scores = [subdata.scores.get(p, 0.0) for p in candidates]
        score_gap = (scores[0] - scores[1]) if len(scores) > 1 else 1.0

        # Decision logic
        if ocr_candidates >= 1 or visual_only_candidates >= 1 or mostly_img_candidates >= 1 or has_vis_term:
            # If query specifically targets tabular structure and Chandra endpoint is configured, route to enrich
            if self.has_chandra and re.search(r"\b(table|tables|tableau|tableaux|column|columns)\b", query, re.IGNORECASE):
                return "enrich", {
                    "reason": "tabular_structure_enrichment",
                    "ocr_candidates": ocr_candidates,
                    "has_visual_terms": has_vis_term
                }
            return "visual", {
                "reason": "visual_query_or_scanned_page",
                "ocr_candidates": ocr_candidates,
                "mostly_img": mostly_img_candidates,
                "has_visual_terms": has_vis_term,
                "score_gap": round(score_gap, 4)
            }

        # Otherwise, candidate pages are clean native digital text
        return "light", {
            "reason": "clean_digital_text",
            "score_gap": round(score_gap, 4),
            "ocr_candidates": 0
        }

    def describe(self):
        return f"rule:{self.argument}"

