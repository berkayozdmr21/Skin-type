"""
app.py
------
Cilt analizi prototipini masaustunde calistiran, tarayici tabanli bir
arayuz (Gradio). Kod tamamen yerel calisir - fotograflariniz hicbir
yere gonderilmez.

Ilk calistirmada 68 nokta yuz landmark modeli (~54 MB) otomatik olarak
GitHub'dan indirilir (internet gerekir, sadece ilk seferde).

Calistirma:
    pip install -r requirements.txt
    python3 app.py

Tarayicida otomatik acilir: http://127.0.0.1:7860
"""

import os
import traceback
import urllib.request

import cv2
import gradio as gr
import numpy as np

from skin_analysis import FaceAnalyzer, analyze_multi
from demo import draw_overlay, LABELS, ATTR_COLORS

MAX_SIDE_PX = 1600  # cok buyuk telefon fotograflarinda (3 foto x sato/Laplacian) sureyi makul tutar

LBF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "lbfmodel.yaml")
LBF_MODEL_URL = (
    "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
)

_analyzer = None


def _ensure_model():
    """Landmark modelini yoksa indirir (sadece ilk calistirmada)."""
    if os.path.exists(LBF_MODEL_PATH) and os.path.getsize(LBF_MODEL_PATH) > 1_000_000:
        return
    print("Yuz landmark modeli indiriliyor (~54 MB, sadece ilk seferde)...")
    urllib.request.urlretrieve(LBF_MODEL_URL, LBF_MODEL_PATH)
    print("Model indirildi:", LBF_MODEL_PATH)


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _ensure_model()
        _analyzer = FaceAnalyzer(lbf_model_path=LBF_MODEL_PATH)
    return _analyzer


_STATUS_TEXT = {
    "ok": ("✅", "kullanıldı"),
    "low_turn": ("⚠️", "yeterince dönmemiş görünüyor (yine de kullanıldı)"),
    "no_face": ("⚠️", "yüz algılanamadı — hesaba katılmadı (biraz daha az dönüp tekrar deneyin)"),
    "not_provided": ("➖", "yüklenmedi — hesaba katılmadı"),
}
_VIEW_NAMES = {"front": "Karşıdan", "right": "Sağa dönük", "left": "Sola dönük"}
_VIEW_SHORT = {"front": "Ön", "right": "Sağ", "left": "Sol"}


def _score_table_html(multi):
    scores = multi["scores"]
    per_view = multi["per_view_scores"]
    quality = multi["front"]["quality_check"]

    def bar(key, label, value, neutral=False, note=None):
        if value is None:
            return f"<div style='margin:6px 0'><b>{label}</b>: veri yok</div>"
        # "Sikilik (deneysel)" gercek bir olcum degil, kaba bir geometrik heuristik -
        # kirmizi/yesil skalasina sokmak yanlis bir "kesinlik" izlenimi veriyordu.
        color = "#999" if neutral else f"rgb({int(255*value/100)},{int(255*(1-value/100))},40)"
        b, g, r = ATTR_COLORS[key]
        dot = f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:rgb({r},{g},{b});margin-right:6px'></span>"
        note_html = f"<div style='font-size:11px;color:#888;margin-top:2px'>{note}</div>" if note else ""
        # pozlara gore kirilim: hangi fotografin hangi degeri verdigi
        parts = [f"{_VIEW_SHORT[v]}: {per_view[v][key]:.0f}" for v in ("front", "right", "left")
                 if v in per_view and per_view[v].get(key) is not None]
        breakdown = f"<div style='font-size:11px;color:#888;margin-top:2px'>{' · '.join(parts)}</div>" if len(parts) > 1 else ""
        return f"""
        <div style='margin:10px 0'>
          <div style='display:flex;justify-content:space-between;font-size:14px'>
            <span>{dot}<b>{label}</b></span><span>{value:.0f}/100</span>
          </div>
          <div style='background:#2a2a2a;border-radius:6px;height:10px;overflow:hidden'>
            <div style='background:{color};width:{value:.0f}%;height:100%'></div>
          </div>
          {breakdown}{note_html}
        </div>"""

    rows = "".join(
        bar(
            key, label, scores.get(key),
            neutral=(key == "sikilik_deneysel"),
            note="(kaba geometrik tahmin, dermatolojik anlamı yok — ileride eğitilmiş modelle değiştirilecek)" if key == "sikilik_deneysel" else None,
        )
        for key, label in LABELS.items()
    )

    def qline(name, info):
        ok = info.get("ok")
        icon = "✅" if ok else "⚠️"
        return f"<div style='margin:4px 0'>{icon} <b>{name}</b>: {info.get('reason','-')}</div>"

    qrows = "".join(
        qline(n, quality[k]) for n, k in
        [("Aydınlatma", "aydinlatma"), ("Poz", "poz"), ("Pozisyon", "pozisyon")]
    )

    srows = "".join(
        f"<div style='margin:4px 0'>{_STATUS_TEXT[multi['status'][v]][0]} <b>{_VIEW_NAMES[v]}</b>: {_STATUS_TEXT[multi['status'][v]][1]}</div>"
        for v in ("front", "right", "left")
    )

    return f"""
    <div style='font-family:sans-serif;color:#eee'>
      <h3>Kullanılan Fotoğraflar</h3>
      {srows}
      <p style='font-size:11px;color:#999'>Skorlar; karşıdan fotoğraf (ağırlık 2) ve yan fotoğraflardan (ağırlık 1) yalnızca kameraya dönük yanağın/gözün ölçümlerinin ağırlıklı ortalamasıdır. Görseldeki işaretler karşıdan fotoğraftandır.</p>
      <h3>Kamera Kalite Kontrolü <span style='font-weight:normal;font-size:12px'>(karşıdan fotoğraf)</span></h3>
      {qrows}
      <h3 style='margin-top:16px'>Cilt Skorları <span style='font-weight:normal;font-size:12px'>(yüksek = daha belirgin sorun)</span></h3>
      {rows}
      <p style='font-size:11px;color:#999;margin-top:14px'>
        Görseldeki renkli noktalar/çizgiler her niteliğin en belirgin aday
        bölgelerini gösterir; nokta sayısı veya büyüklüğü şiddet değildir,
        şiddet skordur. Bu skorlar klasik görüntü işleme ile hesaplanan başlangıç
        kalibrasyonlarıdır — dermatolojik bir teşhis değildir.
      </p>
    </div>"""


def _rgb_to_bgr(arr):
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _load_bgr(image_input):
    """Gradio surumune gore gelen goruntu inputunu BGR numpy array'e cevirir."""
    if image_input is None:
        return None
    if isinstance(image_input, str):
        return cv2.imread(image_input)
    if isinstance(image_input, dict):
        for key in ("path", "image", "composite", "background"):
            val = image_input.get(key)
            if isinstance(val, str):
                return cv2.imread(val)
            if isinstance(val, np.ndarray):
                return _rgb_to_bgr(val)
        return None
    if isinstance(image_input, np.ndarray):
        return _rgb_to_bgr(image_input)
    return None


def _limit_size(image_bgr):
    if image_bgr is None:
        return None
    h, w = image_bgr.shape[:2]
    scale = MAX_SIDE_PX / max(h, w)
    if scale >= 1:
        return image_bgr
    return cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def run_analysis(front_img, right_img, left_img):
    try:
        front_bgr = _limit_size(_load_bgr(front_img))
        if front_bgr is None:
            return None, "<i>Önce tam karşıdan çekilmiş bir fotoğraf yükleyin (sağ/sol dönük fotoğraflar opsiyonel).</i>"
        right_bgr = _limit_size(_load_bgr(right_img))
        left_bgr = _limit_size(_load_bgr(left_img))

        analyzer = _get_analyzer()
        multi = analyze_multi(front_bgr, analyzer, right_bgr, left_bgr)

        if "error" in multi:
            return None, "<b style='color:#e55'>Karşıdan fotoğrafta yüz bulunamadı — daha net, tek yüz içeren, tam karşıdan bir fotoğraf deneyin.</b>"

        canvas_bgr = draw_overlay(front_bgr, multi["front"], scores=multi["scores"])
        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        html = _score_table_html(multi)
        return canvas_rgb, html
    except Exception as e:
        traceback.print_exc()
        return None, f"<b style='color:#e55'>Hata: {e}</b>"


with gr.Blocks(title="Cilt Analizi — Masaüstü Test Arayüzü", theme=gr.themes.Soft()) as demo_app:
    gr.Markdown(
        "# 🔬 Cilt Analizi — Masaüstü Test Arayüzü\n"
        "Klasik görüntü işleme (OpenCV) tabanlı prototip. Tamamen yerel çalışır, "
        "hiçbir fotoğraf internete gönderilmez.\n\n"
        "**Nasıl kullanılır:** 1) tam karşıdan, 2) başınızı hafifçe sağa, 3) hafifçe sola "
        "çevirerek (yaklaşık 30-40°, yüzünüz hâlâ görünsün) üç fotoğraf yükleyin ya da "
        "kamerayla çekin, sonra *Analiz Et*'e basın. Karşıdan fotoğraf zorunlu, yan "
        "fotoğraflar opsiyoneldir — ne kadar çoksa skorlar o kadar güvenilir olur. "
        "Sonuç görseli karşıdan fotoğraf üzerinde çizilir; skorlara yan fotoğraflar da katılır."
    )
    with gr.Row():
        front_in = gr.Image(label="1) Tam karşıdan", sources=["upload", "webcam"], type="numpy")
        right_in = gr.Image(label="2) Başı hafifçe sağa çevrik", sources=["upload", "webcam"], type="numpy")
        left_in = gr.Image(label="3) Başı hafifçe sola çevrik", sources=["upload", "webcam"], type="numpy")
    btn = gr.Button("Analiz Et", variant="primary")
    with gr.Row():
        with gr.Column():
            img_out = gr.Image(label="Sonuç (karşıdan fotoğraf üzerinde)")
        with gr.Column():
            html_out = gr.HTML()

    btn.click(fn=run_analysis, inputs=[front_in, right_in, left_in], outputs=[img_out, html_out])

if __name__ == "__main__":
    demo_app.launch(inbrowser=True)
