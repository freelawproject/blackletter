"""Test opinion pairing against real scan data (A3d vol. 217).

Expected values are hand-verified. Edit EXPECTED_OPINIONS to fix any
pairing that is wrong — the test will catch regressions.
"""

from pathlib import Path

import fitz
import pytest

from blackletter.api import pair as bl_pair

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DET_PATH = FIXTURE_DIR / "a3d_217_detections.json"
PDF_PAGE_COUNT = 1374

# (caption_page_index, key_page_index, first_page_number, last_page_number)
# Hand-verify these values and update as needed.
EXPECTED_OPINIONS = [
    #  idx  caption_pi  key_pi  first_pn  last_pn
    (0, 0, 8, 1, 9),
    (1, 8, 18, 9, 19),
    (2, 19, 39, 20, 40),
    (3, 39, 42, 40, 43),
    (4, 42, 48, 43, 49),
    (5, 48, 55, 49, 56),
    (6, 55, 62, 56, 63),
    (7, 62, 67, 63, 68),
    (8, 67, 75, 68, 76),
    (9, 75, 92, 76, 93),
    (10, 92, 102, 93, 103),
    (11, 102, 121, 103, 122),
    (12, 121, 140, 122, 141),
    (13, 140, 154, 141, 155),
    (14, 154, 171, 155, 172),
    (15, 171, 186, 172, 187),
    (16, 186, 210, 187, 211),
    (17, 210, 231, 211, 232),
    (18, 231, 242, 232, 243),
    (19, 242, 252, 243, 253),
    (20, 252, 265, 253, 266),
    (21, 265, 273, 266, 274),
    (22, 273, 283, 274, 284),
    (23, 283, 295, 284, 296),
    (24, 295, 301, 296, 302),
    (25, 301, 311, 302, 312),
    (26, 312, 321, 313, 322),
    (27, 321, 336, 322, 337),
    (28, 336, 351, 337, 352),
    (29, 351, 357, 352, 358),
    (30, 357, 361, 358, 362),
    (31, 361, 380, 362, 381),
    (32, 380, 395, 381, 396),
    (33, 395, 403, 396, 404),
    (34, 403, 414, 404, 415),
    (35, 414, 423, 415, 424),
    (36, 423, 436, 424, 437),
    (37, 436, 443, 437, 444),
    (38, 443, 484, 444, 485),
    (39, 484, 495, 485, 496),
    (40, 495, 509, 496, 510),
    (41, 509, 520, 510, 521),
    (42, 520, 526, 521, 527),
    (43, 526, 533, 527, 534),
    (44, 533, 539, 534, 540),
    (45, 539, 548, 540, 549),
    (46, 548, 552, 549, 553),
    (47, 552, 557, 553, 558),
    (48, 557, 572, 558, 573),
    (49, 572, 585, 573, 586),
    (50, 585, 592, 586, 593),
    (51, 593, 597, 594, 598),
    (52, 597, 603, 598, 604),
    (53, 603, 625, 604, 626),
    (54, 626, 626, 627, 627),
    (55, 626, 629, 627, 630),
    (56, 629, 632, 630, 633),
    (57, 632, 637, 633, 638),
    (58, 637, 643, 638, 644),
    (59, 644, 649, 645, 650),
    (60, 650, 654, 651, 655),
    (61, 654, 656, 655, 657),
    (62, 656, 669, 657, 670),
    (63, 669, 675, 670, 676),
    (64, 676, 687, 685, 696),
    (65, 687, 693, 696, 702),
    (66, 694, 707, 703, 716),
    (67, 707, 712, 716, 721),
    (68, 712, 728, 721, 737),
    (69, 729, 743, 738, 752),
    (70, 743, 748, 752, 757),
    (71, 748, 794, 757, 803),
    (72, 795, 805, 804, 814),
    (73, 805, 812, 814, 821),
    (74, 813, 818, 822, 827),
    (75, 818, 829, 827, 838),
    (76, 829, 835, 838, 844),
    (77, 835, 843, 844, 852),
    (78, 843, 849, 852, 858),
    (79, 849, 869, 858, 878),
    (80, 869, 869, 878, 878),
    (81, 869, 878, 878, 887),
    (82, 878, 950, 887, 959),
    (83, 951, 1050, 960, 1059),
    (84, 1050, 1055, 1059, 1064),
    (85, 1055, 1083, 1064, 1092),
    (86, 1083, 1093, 1092, 1102),
    (87, 1093, 1102, 1102, 1111),
    (88, 1103, 1107, 1112, 1116),
    (89, 1107, 1115, 1116, 1124),
    (90, 1115, 1129, 1124, 1138),
    (91, 1129, 1139, 1138, 1148),
    (92, 1139, 1150, 1148, 1159),
    (93, 1150, 1167, 1159, 1176),
    (94, 1167, 1174, 1176, 1183),
    (95, 1174, 1183, 1183, 1192),
    (96, 1183, 1201, 1192, 1210),
    (97, 1201, 1219, 1210, 1228),
    (98, 1219, 1236, 1228, 1245),
    (99, 1236, 1261, 1245, 1270),
    (100, 1261, 1270, 1270, 1279),
    (101, 1270, 1295, 1279, 1304),
    (102, 1295, 1310, 1304, 1319),
    (103, 1310, 1323, 1319, 1332),
    (104, 1323, 1332, 1332, 1341),
    (105, 1332, 1343, 1341, 1352),
    (106, 1343, 1353, 1352, 1362),
    (107, 1353, 1361, 1362, 1370),
    (108, 1361, 1373, 1370, 1382),
]


class TestPairingA3d217:
    """Verify opinion pairing against hand-verified expected values."""

    @pytest.fixture(scope="class")
    def opinions(self, tmp_path_factory):
        # Build a stub PDF with the right page count and dimensions
        doc = fitz.open()
        for _ in range(PDF_PAGE_COUNT):
            doc.new_page(width=434.4, height=697.68)
        tmp = tmp_path_factory.mktemp("pdf") / "stub.pdf"
        doc.save(str(tmp))
        doc.close()
        return bl_pair(
            str(DET_PATH),
            str(tmp),
            reporter="a3d",
            volume="217",
            first_page=1,
        )

    def test_opinion_count(self, opinions):
        assert len(opinions) == len(EXPECTED_OPINIONS), (
            f"Expected {len(EXPECTED_OPINIONS)} opinions, got {len(opinions)}"
        )

    def test_caption_and_key_pages(self, opinions):
        errors = []
        for idx, cap_pi, key_pi, first_pn, last_pn in EXPECTED_OPINIONS:
            if idx >= len(opinions):
                errors.append(f"Opinion {idx}: missing (only {len(opinions)} opinions)")
                continue
            op = opinions[idx]
            if op["caption_page"] != cap_pi:
                errors.append(
                    f"Opinion {idx}: caption_page={op['caption_page']}, expected {cap_pi}"
                )
            if op["key_page"] != key_pi:
                errors.append(f"Opinion {idx}: key_page={op['key_page']}, expected {key_pi}")
        assert not errors, "Pairing mismatches:\n" + "\n".join(errors)

    def test_page_numbers(self, opinions):
        errors = []
        for idx, cap_pi, key_pi, first_pn, last_pn in EXPECTED_OPINIONS:
            if idx >= len(opinions):
                continue
            op = opinions[idx]
            actual_first = op.get("first_page_number")
            actual_last = op.get("last_page_number")
            if actual_first != first_pn:
                errors.append(
                    f"Opinion {idx} (cap_pi={cap_pi}): first_page_number={actual_first}, expected {first_pn}"
                )
            if actual_last != last_pn:
                errors.append(
                    f"Opinion {idx} (cap_pi={cap_pi}): last_page_number={actual_last}, expected {last_pn}"
                )
        assert not errors, "Page number mismatches:\n" + "\n".join(errors)

    def test_no_backward_ranges(self, opinions):
        """No opinion should have end_page < caption_page."""
        errors = []
        for i, op in enumerate(opinions):
            ep = op.get("end_page", op["key_page"])
            if ep < op["caption_page"]:
                errors.append(
                    f"Opinion {i}: backward range caption_page={op['caption_page']} > end_page={ep}"
                )
        assert not errors, "Backward ranges found:\n" + "\n".join(errors)

    def test_end_page_equals_key_page(self, opinions):
        """Every opinion's end_page must equal its key_page (caption-to-key range)."""
        errors = []
        for i, op in enumerate(opinions):
            if op["end_page"] != op["key_page"]:
                errors.append(
                    f"Opinion {i}: end_page={op['end_page']} != key_page={op['key_page']}"
                )
        assert not errors, "end_page/key_page mismatches:\n" + "\n".join(errors)

    def test_page_count_matches_range(self, opinions):
        """page_count must equal key_page - caption_page + 1."""
        errors = []
        for i, op in enumerate(opinions):
            expected = op["key_page"] - op["caption_page"] + 1
            if op["page_count"] != expected:
                errors.append(
                    f"Opinion {i}: page_count={op['page_count']}, "
                    f"expected {expected} (caption={op['caption_page']}, key={op['key_page']})"
                )
        assert not errors, "page_count mismatches:\n" + "\n".join(errors)

    def test_no_giant_opinions(self, opinions):
        """Flag opinions spanning more than 50 pages as likely missed splits."""
        # These are verified as legitimately large opinions
        known_large = {82, 83}
        large = []
        for i, op in enumerate(opinions):
            pc = op.get("page_count", 1)
            if pc > 50 and i not in known_large:
                large.append(
                    f"Opinion {i}: {pc} pages (caption_page={op['caption_page']}, "
                    f"pages={op.get('first_page_number')}-{op.get('last_page_number')})"
                )
        assert not large, "Suspiciously large opinions:\n" + "\n".join(large)
