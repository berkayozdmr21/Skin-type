"""
demo.py
-------
skin_analysis.py pipeline'ini bir fotograf uzerinde calistirir, sonucu
gorsellestirir (ROI'ler, cene hatti, kizariklik isi haritasi, skor paneli)
ve JSON olarak skorlari yazdirir.

Kullanim:
    python3 demo.py <girdi_fotograf.jpg> <cikti_fotograf.jpg>
"""

import sys
import json
import cv2
import numpy as np

from skin_analysis import FaceAnalyzer, analyze_image, _mask_from_polygon

LABELS = {
    "kizariklik": "Kizariklik",
    "parlama": "Parlama / Yaglilik",
    "gozenek": "Gozenek / Doku",
    "kirisiklik": "Kirisiklik",
    "lekelenme": "Lekelenme",
    "gozalti": "Goz alti (morluk)",
    "sikilik_deneysel": "Sikilik (deneysel)",
}


# Her nitelik icin ayri isaretleme rengi (BGR)
ATTR_COLORS = {
    "kizariklik": (0, 0, 255),         # kirmizi
    "parlama": (0, 235, 255),          # sari
    "gozenek": (255, 220, 0),          # cyan
    "kirisiklik": (80, 220, 80),       # yesil
    "lekelenme": (0, 128, 255),        # turuncu
    "gozalti": (220, 60, 220),         # mor/magenta
    "sikilik_deneysel": (150, 150, 150),  # gri (anlamli bir olcum degil)
}


def _blob_dots(mask, max_n, min_area):
    """Bool maskeyi bagli bilesenlere ayirip en buyuk max_n tanesinin merkezini dondurur."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    order = sorted(
        (i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area),
        key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True,
    )[:max_n]
    return [(cents[i], stats[i, cv2.CC_STAT_AREA]) for i in order]


def _draw_dots(img, blobs, color, base_r):
    for (cx, cy), area in blobs:
        r = int(np.clip(np.sqrt(area / np.pi) + base_r, 2, 4 * base_r))
        c = (int(round(cx)), int(round(cy)))
        cv2.circle(img, c, r + 1, (20, 20, 20), 1, lineType=cv2.LINE_AA)  # koyu halka: ten uzerinde okunaklilik
        cv2.circle(img, c, r, color, -1, lineType=cv2.LINE_AA)


def draw_overlay(image_bgr, result, scores=None):
    """result: karsidan fotonun analiz_image() sonucu (ROI/isaretler bundan cizilir).
    scores: verilirse (cok-poz birlesik skorlar) panelde ve kizariklik yogunlugunda
    bunlar kullanilir; verilmezse result['scores']."""
    out = image_bgr.copy()
    landmarks = result["landmarks"]
    rois = result["rois"]
    marks = result.get("marks", {})
    scores = scores if scores is not None else result["scores"]
    face_w = int(result["bbox"][2])
    base_r = max(2, int(round(face_w / 150)))

    # 1) Kizariklik isi haritasi (yanaklar) - opaklik skorla orantili.
    # Skor dusukken katman neredeyse gorunmez kalsin, skor yukseldikce belirginlessin -
    # sabit alpha, dusuk skorlu yuzlerin de "asiri kirmizi" gorunmesine sebep oluyordu.
    cheek_mask = cv2.bitwise_or(
        _mask_from_polygon(out.shape, rois["left_cheek"]),
        _mask_from_polygon(out.shape, rois["right_cheek"]),
    )
    redness_score = scores.get("kizariklik")
    redness_frac = (redness_score / 100) if redness_score is not None else 0.0
    heat = np.zeros_like(out)
    heat[:, :] = (0, 0, 255)  # BGR kirmizi
    alpha_mask = (cheek_mask > 0).astype(np.float32) * (0.05 + 0.30 * redness_frac)
    for c in range(3):
        out[:, :, c] = (out[:, :, c] * (1 - alpha_mask) + heat[:, :, c] * alpha_mask).astype(np.uint8)

    # 1b) Goz alti: yari saydam mor dolgu + ince mor kontur
    under_eye = marks.get("gozalti")
    if under_eye is not None:
        ue = under_eye.astype(np.uint8)
        tint = np.array(ATTR_COLORS["gozalti"], dtype=np.float32)
        sel = ue > 0
        out[sel] = (out[sel] * 0.7 + tint * 0.3).astype(np.uint8)
        cnts, _ = cv2.findContours(ue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, ATTR_COLORS["gozalti"], 1, lineType=cv2.LINE_AA)

    # 2) ROI konturlari - tum ROI'ler icin tek, notr, soft renk (alpha-blend ile
    # inceltilmis gorunum; cv2.polylines 1px altina inemedigi icin opacity dusurulur).
    contour_overlay = out.copy()
    contour_color = (210, 210, 210)  # BGR acik gri
    roi_names_for_contour = ["forehead", "nose", "left_cheek", "right_cheek"]
    for name in roi_names_for_contour:
        pts = rois[name].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(contour_overlay, [pts], isClosed=True, color=contour_color, thickness=1, lineType=cv2.LINE_AA)

    # 3) Cene hatti (kontur cizgisi) - ince, yari saydam beyaz
    jaw = rois["jaw_line"].astype(np.int32)
    cv2.polylines(contour_overlay, [jaw.reshape(-1, 1, 2)], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(contour_overlay, 0.5, out, 0.5, 0)

    # 4) Nitelik isaretleri - her nitelik kendi renginde, nokta nokta.
    # Kirisiklik: en belirgin cizgi pikselleri (yesil, ince)
    wrinkles = marks.get("kirisiklik")
    if wrinkles is not None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(wrinkles.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 8  # tek-piksel gurultuyu ele, cizgi parcalarini tut
        wr = keep[labels]
        col = np.array(ATTR_COLORS["kirisiklik"], dtype=np.float32)
        out[wr] = (out[wr] * 0.3 + col * 0.7).astype(np.uint8)
    # Gozenek / lekelenme / parlama: bagli bilesen merkezlerinde nokta. Cok sayidaki
    # kucuk gozenek noktalari once, az sayidaki belirgin lekeler/parlamalar ustune cizilir.
    spot_min_area = max(6, int((face_w / 80) ** 2))
    for key, max_n, min_area in (
        ("gozenek", 150, 2),
        ("lekelenme", 80, spot_min_area),
        ("parlama", 60, 4),
    ):
        m = marks.get(key)
        if m is not None:
            _draw_dots(out, _blob_dots(m, max_n, min_area), ATTR_COLORS[key], base_r)

    # 5) Landmark noktalari - sadece cene hatti (0-16); kas/goz/burun/agiz noktalari
    # nitelik isaretleriyle karismasin ve gozlukle cakismasin diye cizilmez.
    for (x, y) in landmarks[:17].astype(int):
        cv2.circle(out, (x, y), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # 6) Skor paneli (sag taraf) - renkli daire = gorseldeki isaret rengi (lejant)
    panel_w, line_h = 340, 34
    n_lines = len(LABELS) + 1
    panel_h = line_h * n_lines + 20
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (30, 24, 20)

    y0 = 26
    # Sikilik deneysel/anlamsiz bir olcum - genel skora katilmaz
    overall_vals = [v for k, v in scores.items() if v is not None and k != "sikilik_deneysel"]
    overall = sum(overall_vals) / len(overall_vals) if overall_vals else None
    cv2.putText(panel, f"Genel skor: {overall:.0f}/100" if overall is not None else "Genel skor: -",
                (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    y0 += line_h
    for key, label in LABELS.items():
        v = scores.get(key)
        txt = f"{label}: {v:.0f}/100" if v is not None else f"{label}: -"
        cv2.circle(panel, (22, y0 - 6), 6, ATTR_COLORS[key], -1, lineType=cv2.LINE_AA)
        cv2.putText(panel, txt, (38, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y0 += line_h

    # panel + goruntuyu yan yana birlestir
    H = max(out.shape[0], panel.shape[0])
    canvas = np.zeros((H, out.shape[1] + panel_w, 3), dtype=np.uint8)
    canvas[: out.shape[0], : out.shape[1]] = out
    canvas[: panel.shape[0], out.shape[1]:] = panel
    return canvas


def main():
    if len(sys.argv) != 3:
        print("Kullanim: python3 demo.py <girdi.jpg> <cikti.jpg>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    image = cv2.imread(in_path)
    if image is None:
        print(f"Goruntu okunamadi: {in_path}")
        sys.exit(1)

    analyzer = FaceAnalyzer(lbf_model_path="lbfmodel.yaml")
    result = analyze_image(image, analyzer)

    if "error" in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(1)

    canvas = draw_overlay(image, result)
    cv2.imwrite(out_path, canvas)

    printable = {
        "skorlar": result["scores"],
        "kalite_kontrol": {
            k: {kk: vv for kk, vv in v.items() if kk != "pose"}
            for k, v in result["quality_check"].items()
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"\nGorsellestirilmis cikti: {out_path}")


if __name__ == "__main__":
    main()
