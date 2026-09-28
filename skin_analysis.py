"""
skin_analysis.py
-----------------
Klasik goruntu isleme (OpenCV) + yuz landmark tespiti (68 nokta, LBF) kullanarak
cilt analizi prototipi.

Mimari (onceki mesajda tartisilan hibrit yaklasimin uygulamasi):

  1) Yuz tespiti (Haar cascade, bundled - internet gerekmez)
  2) 68 nokta landmark tespiti (OpenCV cv2.face LBF facemark)
  3) Landmarklardan bolgesel ROI cikarimi (alin, yanaklar, T-bolgesi, goz alti, cene hatti)
  4) Her ROI icin klasik goruntu isleme tabanli skor hesaplama:
       - kizariklik (Lab a* kanali)
       - parlama/yaglilik (HSV V kanali - specular highlight orani)
       - gozenek/doku (Laplacian varyansi)
       - kirisiklik (skimage sato/frangi ridge filtresi)
       - lekelenme (yerel parlaklik sapmasi - adaptif esikleme)
       - goz alti morluk/torba (yerel renk/parlaklik farki)
       - sikilik (deneysel - cene hatti egrilik heuristigi, DUSUK GUVENILIRLIK)
  5) Isik/poz kontrolu (canli kamera rehberligi icin: aydinlatma, duz bakis, pozisyon)
  6) Akne / cilt tipi (yagli-kuru-normal) icin: bu adim klasik CV ile guvenilir
     yapilamiyor - burada bir "plug-in" fonksiyon olarak birakildi
     (bkz. classify_skin_type_and_acne). Onceki mesajda bulunan acik veri
     setleriyle egitilecek transfer-learning modeli bu fonksiyonun icine
     yerlestirilecek. Simdilik bu fonksiyon None döner.

NOT: Buradaki esik degerleri / kalibrasyon sabitleri, dermatolog onayli
gercek olcumlerle degil, literatur + gozlemle belirlenmis BASLANGIC
degerleridir (bkz. README.md, "Kalibrasyon" bolumu). Ekibin kendi
etiketli ornekleriyle yeniden kalibre edilmesi gerekir.
"""

import cv2
import numpy as np
from skimage.filters import sato

# ---------------------------------------------------------------------------
# 1-2) Yuz tespiti ve 68 nokta landmark
# ---------------------------------------------------------------------------

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceAnalyzer:
    def __init__(self, lbf_model_path="lbfmodel.yaml"):
        self.face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(lbf_model_path)

    def detect(self, image_bgr, min_neighbors=6):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(120, 120)
        )
        if len(faces) == 0:
            return None
        # en buyuk yuzu al
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        bbox = faces[0]
        ok, landmarks = self.facemark.fit(gray, np.array([bbox]))
        if not ok:
            return None
        pts = landmarks[0][0].astype(np.float64)  # (68, 2)
        return {"bbox": bbox, "landmarks": pts}


# ---------------------------------------------------------------------------
# 3) ROI (bolge) cikarimi
# ---------------------------------------------------------------------------

def _poly(*idx_groups, landmarks):
    """Landmark indekslerinden bir poligon (Nx2 int32) olusturur."""
    pts = np.concatenate([landmarks[list(g)] for g in idx_groups], axis=0)
    return pts.astype(np.int32)


def build_rois(landmarks, bbox):
    """68 noktadan bolgesel ROI poligonlari uretir. Heuristik geometri - kesin
    anatomik sinir degil, MVP icin yeterli yaklastirma."""
    x, y, w, h = bbox
    jaw = landmarks[0:17]
    reyebrow = landmarks[17:22]
    leyebrow = landmarks[22:27]
    nose = landmarks[27:36]
    reye = landmarks[36:42]
    leye = landmarks[42:48]
    mouth = landmarks[48:68]

    eyebrow_top_y = min(reyebrow[:, 1].min(), leyebrow[:, 1].min())
    # sac cizgisi tahmini: kas-cene mesafesinin ~0.9'u kadar yukarida
    chin_y = jaw[8, 1]
    upper_face_h = chin_y - eyebrow_top_y
    hairline_y = max(y, eyebrow_top_y - 0.85 * upper_face_h)

    forehead = np.array([
        [jaw[0, 0], eyebrow_top_y],
        [jaw[16, 0], eyebrow_top_y],
        [jaw[16, 0], hairline_y],
        [jaw[0, 0], hairline_y],
    ], dtype=np.int32)

    # sag yanak (goruntude sol taraf): kas baslangici - goz - burun kenari - cene
    right_cheek = np.array([
        reyebrow[0], reye[0], nose[4], mouth[0], jaw[3], jaw[1],
    ], dtype=np.int32)

    left_cheek = np.array([
        leyebrow[-1], leye[3], nose[8], mouth[6], jaw[13], jaw[15],
    ], dtype=np.int32)

    nose = _poly(range(27, 36), landmarks=landmarks)  # burun (ayri poligon; T-bolgesi
    # maskesi analyze_image() icinde forehead_mask | nose_mask olarak OR'lanir -
    # tek bir poligon olarak birlestirilmez, aksi halde iki ayri bolge arasinda
    # sahte bir "kelebek" kontur cizgisi olusur.

    def under_eye(eye_pts, drop_ratio=0.55):
        # Bandi alt kapak/kirpik sinirindan (eye_pts[4], eye_pts[5]) baslatiyoruz,
        # goz kosesi hizasindan degil - aksi halde kirpik/goz ici gibi cok koyu
        # pikseller ortalamaya karisip skoru yapay olarak sisiriyor.
        eye_w = np.linalg.norm(eye_pts[3] - eye_pts[0])
        offset = eye_w * drop_ratio
        margin = eye_w * 0.08  # kirpik cizgisinden kucuk bir bosluk birak
        lower = eye_pts[[4, 5]]
        left_lower, right_lower = sorted(lower, key=lambda p: p[0])
        top_l = [left_lower[0], left_lower[1] + margin]
        top_r = [right_lower[0], right_lower[1] + margin]
        bot_l = [left_lower[0], left_lower[1] + margin + offset]
        bot_r = [right_lower[0], right_lower[1] + margin + offset]
        band = np.array([top_l, top_r, bot_r, bot_l], dtype=np.int32)
        return band

    r_under_eye = under_eye(reye)
    l_under_eye = under_eye(leye)

    return {
        "forehead": forehead,
        "right_cheek": right_cheek,
        "left_cheek": left_cheek,
        "nose": nose,
        "right_under_eye": r_under_eye,
        "left_under_eye": l_under_eye,
        "jaw_line": jaw,
    }


def _mask_from_polygon(shape, polygon):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


# ---------------------------------------------------------------------------
# 4) Klasik goruntu isleme skorlari (0-100, yuksek = sorun daha belirgin)
# ---------------------------------------------------------------------------

def _clip01_to_100(v, lo, hi):
    v = (v - lo) / (hi - lo + 1e-6)
    return float(np.clip(v, 0, 1) * 100)


def score_redness(image_bgr, mask):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    vals = a[mask > 0]
    if vals.size == 0:
        return None
    a_mean = float(vals.mean())
    # Lab a*: notr=128; cilt tonu tipik olarak ~132-146; eritem/kizariklik ile yukselir.
    # Kaynak: Tao et al. 2023 (VISIA), Logger et al. - a* ile eritem korelasyonu.
    return _clip01_to_100(a_mean, 132, 152), a_mean


def score_shine(image_bgr, mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    region = mask > 0
    if region.sum() == 0:
        return None
    # specular (parlama) piksel: cok parlak + dusuk doygunluk
    specular = (v > 235) & (s < 60) & region
    ratio = specular.sum() / region.sum()
    return _clip01_to_100(ratio, 0.0, 0.12), ratio, specular


def score_texture_pores(image_bgr, mask, face_mask):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    vals = lap[mask > 0]
    face_vals = lap[face_mask > 0]
    if vals.size == 0 or face_vals.size == 0:
        return None
    face_var = float(face_vals.var())
    if face_var < 1e-6:
        return None
    # Sabit esik yerine ROI varyansini yuzun genel varyansina orantiliyoruz -
    # boylece fotograf/kamera gurultu seviyesinden bagimsiz, yuze GORELI bir
    # doku olcumu elde ediliyor (ayni telefon farkli isikta tutarli kaliyor).
    ratio = float(vals.var()) / face_var
    # Isaretleme: ROI icindeki en guclu %3 Laplacian tepkisi ("en belirgin doku
    # noktalari"); skorun kendisi yukaridaki orandan gelir, nokta sayisi siddet degildir.
    abs_lap = np.abs(lap)
    marks = (abs_lap > np.percentile(abs_lap[mask > 0], 97)) & (mask > 0)
    return _clip01_to_100(ratio, 0.5, 2.5), ratio, marks


def score_wrinkles(image_bgr, mask):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    ridge = sato(gray, sigmas=range(1, 4), black_ridges=True)
    vals = ridge[mask > 0]
    if vals.size == 0:
        return None
    strength = float(vals.mean())
    marks = (ridge > np.percentile(vals, 92)) & (mask > 0)  # en belirgin cizgi pikselleri
    return _clip01_to_100(strength, 0.0, 0.06), strength, marks


def score_pigmentation(image_bgr, mask, face_mask):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=15)
    detail = blur - L  # pozitif = yerel ortalamadan daha koyu (leke adayi)
    region = mask > 0
    face_region = face_mask > 0
    if region.sum() == 0 or face_region.sum() == 0:
        return None
    # Sabit "detail > 10" esigi yerine, o fotografin kendi L kanali sapmasina
    # gore adaptif esik kullaniyoruz - parlak/dusuk kontrastli fotograflarda
    # esigin sabit kalmasi leke oranini yapay olarak satüre ediyordu.
    face_std = float(L[face_region].std())
    threshold = max(0.8 * face_std, 1e-3)
    dark_spots = (detail > threshold) & region
    ratio = dark_spots.sum() / region.sum()
    return _clip01_to_100(ratio, 0.0, 0.10), ratio, dark_spots


def score_under_eye(image_bgr, eye_mask, cheek_mask):
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    eye_vals = L[eye_mask > 0]
    cheek_vals = L[cheek_mask > 0]
    if eye_vals.size == 0 or cheek_vals.size == 0:
        return None
    delta = float(cheek_vals.mean() - eye_vals.mean())  # pozitif = goz alti daha koyu
    return _clip01_to_100(delta, 0, 45), delta, eye_mask > 0


def score_jaw_firmness_experimental(landmarks):
    """DENEYSEL / DUSUK GUVENILIRLIK: cene hatti egriligini kabaca olcer.
    Gercek 'sikilik' tahmini icin nufus-normalize edilmis egitimli bir model
    gerekir - burada sadece mimarinin tamamlanmasi icin bir yer tutucu."""
    jaw = landmarks[0:17]
    # jaw genisligi / yuz uzunlugu orani - kaba bir sarkma gostergesi degil,
    # sadece geometrik bir referans olcum.
    width = np.linalg.norm(jaw[16] - jaw[0])
    chin = jaw[8]
    top = (jaw[0] + jaw[16]) / 2
    height = np.linalg.norm(chin - top)
    ratio = float(width / (height + 1e-6))
    return _clip01_to_100(ratio, 1.1, 1.9), ratio


# ---------------------------------------------------------------------------
# 5) Isik / poz kontrolu (canli kamera rehberligi icin)
# ---------------------------------------------------------------------------

_MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),        # burun ucu (30)
    (0.0, -330.0, -65.0),   # cene (8)
    (-225.0, 170.0, -135.0),  # sol goz sol kose (36)
    (225.0, 170.0, -135.0),   # sag goz sag kose (45)
    (-150.0, -150.0, -125.0),  # agiz sol kose (48)
    (150.0, -150.0, -125.0),   # agiz sag kose (54)
], dtype=np.float64)


def estimate_head_pose(landmarks, image_shape):
    h, w = image_shape[:2]
    image_points = np.array([
        landmarks[30], landmarks[8], landmarks[36],
        landmarks[45], landmarks[48], landmarks[54],
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}


def check_lighting(image_bgr, face_mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2][face_mask > 0]
    if v.size == 0:
        return {"ok": False, "reason": "yuz bulunamadi"}
    mean_v = float(v.mean())
    std_v = float(v.std())
    if mean_v < 80:
        return {"ok": False, "reason": "isigi_artir", "mean_v": mean_v}
    if mean_v > 210 and std_v < 20:
        return {"ok": False, "reason": "asiri_pozlama", "mean_v": mean_v}
    return {"ok": True, "reason": "iyi", "mean_v": mean_v}


def check_pose(pose):
    if pose is None:
        return {"ok": False, "reason": "poz_hesaplanamadi"}
    if abs(pose["yaw"]) > 15 or abs(pose["pitch"]) > 15:
        return {"ok": False, "reason": "duz_bakin", "pose": pose}
    return {"ok": True, "reason": "iyi", "pose": pose}


def check_face_position(bbox, image_shape):
    h, w = image_shape[:2]
    _, _, fw, fh = bbox
    frame_area = w * h
    face_area = fw * fh
    ratio = face_area / frame_area
    if ratio < 0.12:
        return {"ok": False, "reason": "yaklas", "ratio": ratio}
    if ratio > 0.55:
        return {"ok": False, "reason": "uzaklas", "ratio": ratio}
    return {"ok": True, "reason": "iyi", "ratio": ratio}


# ---------------------------------------------------------------------------
# 6) Egitimli model icin plug-in noktasi (bu depoda dahil degil)
# ---------------------------------------------------------------------------

def classify_skin_type_and_acne(face_crop_bgr):
    """PLACEHOLDER: Onceki mesajdaki yol haritasinin 4. adiminda egitilecek
    transfer-learning modeli (MobileNetV2/EfficientNet-B0, Kaggle Oily-Dry-Normal
    + Roboflow setleriyle fine-tune edilmis) buraya entegre edilecek.

    Klasik goruntu isleme burada guvenilir sonuc vermiyor (bkz. onceki mesajdaki
    fizibilite tablosu) - bu yuzden bilincli olarak bos birakildi.

    Beklenen donus formati:
        {"skin_type": "oily"|"dry"|"normal", "skin_type_confidence": float,
         "acne_severity": 0-100, "acne_confidence": float}
    """
    return None


# ---------------------------------------------------------------------------
# Ana pipeline
# ---------------------------------------------------------------------------

def turn_info(landmarks):
    """Kafanin hangi yana dondugunu geometriden bulur (yaw isaretine guvenmeden).

    Burun ucunun (30) iki yanak kenarina (2, 14) yatay uzakliklari karsilastirilir;
    daha genis gorunen yanak kameraya yakin olandir. 'right' = ekranin solundaki
    (ozne'nin sag) yanak/goz - build_rois'taki right_cheek ile ayni adlandirma.
    Donus: (yakin_taraf, oran) - oran ~1.0 = karsidan, buyudukce daha cok donuk."""
    nose_x = landmarks[30, 0]
    img_left = abs(nose_x - landmarks[2, 0])
    img_right = abs(landmarks[14, 0] - nose_x)
    near = "right" if img_left >= img_right else "left"
    ratio = max(img_left, img_right) / max(min(img_left, img_right), 1e-6)
    return near, float(ratio)


def analyze_image(image_bgr, analyzer: FaceAnalyzer, view="front"):
    """view='front': tum bolgeler. view='side': sadece kameraya yakin yanak/goz +
    alin/burun skorlanir (uzak yanak perspektifte sikisik oldugu icin atlanir)."""
    det = analyzer.detect(image_bgr, min_neighbors=6 if view == "front" else 3)
    if det is None:
        return {"error": "yuz_bulunamadi"}

    landmarks = det["landmarks"]
    bbox = det["bbox"]
    rois = build_rois(landmarks, bbox)
    shape = image_bgr.shape

    masks = {name: _mask_from_polygon(shape, poly) for name, poly in rois.items()
              if name != "jaw_line"}
    face_mask = _mask_from_polygon(shape, np.vstack([landmarks[0:17], landmarks[26], landmarks[17]]))

    near, turn_ratio = turn_info(landmarks)
    if view == "side":
        cheek_mask = masks[f"{near}_cheek"]
        under_eye_mask = masks[f"{near}_under_eye"]
    else:
        cheek_mask = cv2.bitwise_or(masks["left_cheek"], masks["right_cheek"])
        under_eye_mask = cv2.bitwise_or(masks["left_under_eye"], masks["right_under_eye"])
    t_zone_mask = cv2.bitwise_or(masks["forehead"], masks["nose"])

    scores = {}
    scores["kizariklik"] = score_redness(image_bgr, cheek_mask)
    scores["parlama"] = score_shine(image_bgr, t_zone_mask)
    scores["gozenek"] = score_texture_pores(image_bgr, cheek_mask, face_mask)
    scores["kirisiklik"] = score_wrinkles(image_bgr, masks["forehead"])
    scores["lekelenme"] = score_pigmentation(image_bgr, cheek_mask, face_mask)
    scores["gozalti"] = score_under_eye(image_bgr, under_eye_mask, cheek_mask)
    if view == "front":
        scores["sikilik_deneysel"] = score_jaw_firmness_experimental(landmarks)

    quality = {
        "aydinlatma": check_lighting(image_bgr, face_mask),
        "pozisyon": check_face_position(bbox, shape),
    }
    if view == "front":
        quality["poz"] = check_pose(estimate_head_pose(landmarks, shape))

    return {
        "view": view,
        "landmarks": landmarks,
        "bbox": bbox,
        "rois": rois,
        "near_side": near,
        "turn_ratio": turn_ratio,
        "scores": {k: (v[0] if v else None) for k, v in scores.items()},
        "raw_values": {k: (v[1] if v else None) for k, v in scores.items()},
        "marks": {k: v[2] for k, v in scores.items() if v and len(v) > 2},
        "quality_check": quality,
        "skin_type_and_acne": classify_skin_type_and_acne(None),  # placeholder
    }


# ---------------------------------------------------------------------------
# Cok-poz birlestirme (karsidan + saga + sola)
# ---------------------------------------------------------------------------

FRONT_WEIGHT = 2.0   # karsidan poz tum bolgeleri kapsar, daha guvenilir
SIDE_WEIGHT = 1.0    # yan pozlar sadece kendi taraflarini olcer
MIN_SIDE_TURN_RATIO = 1.15  # bunun altinda "yan" foto aslinda neredeyse karsidan


def aggregate_scores(per_view):
    """per_view: {'front': {...skorlar}, 'right': {...}, 'left': {...}} (eksik/None olabilir).
    Her nitelik icin, o niteligi olcebilen pozlarin agirlikli ortalamasi."""
    weights = {"front": FRONT_WEIGHT, "right": SIDE_WEIGHT, "left": SIDE_WEIGHT}
    keys = []
    for s in per_view.values():
        for k in (s or {}):
            if k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        num = den = 0.0
        for view, s in per_view.items():
            v = (s or {}).get(k)
            if v is not None:
                num += weights[view] * v
                den += weights[view]
        out[k] = (num / den) if den else None
    return out


def analyze_multi(front_bgr, analyzer, right_bgr=None, left_bgr=None):
    """Karsidan foto zorunlu; yan fotograflar opsiyonel. Sonuc gorseli karsidan
    fotoya cizilir, skorlar tum pozlardan birlestirilir."""
    front = analyze_image(front_bgr, analyzer, view="front")
    if "error" in front:
        return {"error": front["error"]}

    per_view = {"front": front["scores"]}
    status = {"front": "ok"}
    sides = {}
    for name, img in (("right", right_bgr), ("left", left_bgr)):
        if img is None:
            status[name] = "not_provided"
            continue
        res = analyze_image(img, analyzer, view="side")
        if "error" in res:
            status[name] = "no_face"
            continue
        sides[name] = res
        per_view[name] = res["scores"]
        status[name] = "low_turn" if res["turn_ratio"] < MIN_SIDE_TURN_RATIO else "ok"

    return {
        "front": front,
        "sides": sides,
        "scores": aggregate_scores(per_view),
        "per_view_scores": per_view,
        "status": status,
    }
