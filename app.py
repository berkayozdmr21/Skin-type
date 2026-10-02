"""
app.py
------
Cilt analizi prototipini masaustunde calistiran, tarayici tabanli bir
arayuz (Gradio). Kod tamamen yerel calisir - fotograflariniz hicbir
yere gonderilmez.

Ilk calistirmada 68 nokta yuz landmark modeli (~54 MB) otomatik olarak
GitHub'dan indirilir (internet gerekir, sadece ilk seferde) ve kullanici
profilindeki kalici bir klasore kaydedilir.

Calistirma:
    pip install -r requirements.txt
    python3 app.py

Tarayicida otomatik acilir: http://127.0.0.1:7860
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # .exe genelde ofline acilir, telemetri denemesini atla

import traceback
import urllib.request

import cv2
import gradio as gr
import numpy as np

from skin_analysis import FaceAnalyzer, analyze_multi
from demo import draw_overlay, ATTR_COLORS

MAX_SIDE_PX = 1600  # cok buyuk telefon fotograflarinda (3 foto x sato/Laplacian) sureyi makul tutar


def _app_data_dir():
    """Kalici, kullaniciya ozel bir klasor dondurur. .exe olarak paketlenince
    (PyInstaller) calisma dizini her acilista silinen gecici bir klasor olabilir -
    model dosyasi orada tutulursa her acilista yeniden 54 MB indirilir. Bu yuzden
    script olarak da, exe olarak da hep ayni (kullanici profili altindaki) sabit
    klasor kullanilir."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "CiltAnalizi")
    os.makedirs(path, exist_ok=True)
    return path


LBF_MODEL_PATH = os.path.join(_app_data_dir(), "lbfmodel.yaml")
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

# demo.LABELS, cv2.putText (OpenCV) icin bilerek ASCII tutuluyor - OpenCV'nin yerlesik
# fontu Turkce aksanli karakterleri duzgun cizemiyor. HTML panelinde kisitlama yok,
# bu yuzden burada duzgun Turkce etiketler kullaniliyor.
_LABELS_TR = {
    "kizariklik": "Kızarıklık",
    "parlama": "Parlama / Yağlılık",
    "gozenek": "Gözenek / Doku",
    "kirisiklik": "Kırışıklık",
    "lekelenme": "Lekelenme",
    "gozalti": "Göz altı (morluk)",
    "sikilik_deneysel": "Sıkılık (deneysel)",
}

_REASON_TR = {
    "yuz bulunamadi": "yüz bulunamadı",
    "isigi_artir": "ışığı artırın",
    "asiri_pozlama": "aşırı pozlama, ışığı azaltın",
    "iyi": "iyi",
    "poz_hesaplanamadi": "poz hesaplanamadı",
    "duz_bakin": "kameraya düz bakın",
    "yaklas": "kameraya biraz yaklaşın",
    "uzaklas": "kameradan biraz uzaklaşın",
}


def _score_table_html(multi):
    scores = multi["scores"]
    per_view = multi["per_view_scores"]
    quality = multi["front"]["quality_check"]

    def bar(key, label, value, neutral=False, note=None):
        if value is None:
            return f"<div class='sa-row'><b>{label}</b>: veri yok</div>"
        # "Sikilik (deneysel)" gercek bir olcum degil, kaba bir geometrik heuristik -
        # kirmizi/yesil skalasina sokmak yanlis bir "kesinlik" izlenimi veriyordu.
        color = "#b0b6ba" if neutral else f"rgb({int(255*value/100)},{int(200*(1-value/100))},70)"
        b, g, r = ATTR_COLORS[key]
        dot = f"<span class='sa-dot' style='background:rgb({r},{g},{b})'></span>"
        note_html = f"<div class='sa-note'>{note}</div>" if note else ""
        parts = [f"{_VIEW_SHORT[v]}: {per_view[v][key]:.0f}" for v in ("front", "right", "left")
                 if v in per_view and per_view[v].get(key) is not None]
        breakdown = f"<div class='sa-note'>{' · '.join(parts)}</div>" if len(parts) > 1 else ""
        return f"""
        <div class='sa-row'>
          <div class='sa-row-top'>
            <span>{dot}<b>{label}</b></span><span>{value:.0f}/100</span>
          </div>
          <div class='sa-track'><div class='sa-fill' style='background:{color};width:{value:.0f}%'></div></div>
          {breakdown}{note_html}
        </div>"""

    rows = "".join(
        bar(
            key, label, scores.get(key),
            neutral=(key == "sikilik_deneysel"),
            note="(kaba geometrik tahmin, dermatolojik anlamı yok — ileride eğitilmiş modelle değiştirilecek)" if key == "sikilik_deneysel" else None,
        )
        for key, label in _LABELS_TR.items()
    )

    def qline(name, info):
        ok = info.get("ok")
        icon = "✅" if ok else "⚠️"
        reason = info.get("reason", "-")
        return f"<div class='sa-qline'>{icon} <b>{name}</b>: {_REASON_TR.get(reason, reason)}</div>"

    qrows = "".join(
        qline(n, quality[k]) for n, k in
        [("Aydınlatma", "aydinlatma"), ("Poz", "poz"), ("Pozisyon", "pozisyon")]
    )

    srows = "".join(
        f"<div class='sa-qline'>{_STATUS_TEXT[multi['status'][v]][0]} <b>{_VIEW_NAMES[v]}</b>: {_STATUS_TEXT[multi['status'][v]][1]}</div>"
        for v in ("front", "right", "left")
    )

    return f"""
    <div class='sa-panel'>
      <h4>📸 Kullanılan fotoğraflar</h4>
      {srows}
      <p class='sa-hint'>Skorlar; karşıdan fotoğraf (ağırlık 2) ve yan fotoğraflardan (ağırlık 1) yalnızca kameraya dönük yanağın/gözün ölçümlerinin ağırlıklı ortalamasıdır.</p>
      <h4>💡 Kamera kalite kontrolü</h4>
      {qrows}
      <h4>🧪 Cilt skorları <span class='sa-hint-inline'>(yüksek = daha belirgin sorun)</span></h4>
      {rows}
      <p class='sa-hint'>
        Görseldeki renkli noktalar/çizgiler her niteliğin en belirgin aday bölgelerini
        gösterir; nokta sayısı/büyüklüğü şiddet değildir, şiddet skordur. Bu skorlar
        klasik görüntü işleme ile hesaplanan başlangıç kalibrasyonlarıdır — dermatolojik
        bir teşhis değildir.
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
            return None, "<i>Önce tam karşıdan çekilmiş bir fotoğraf ekleyin (sağ/sol dönük fotoğraflar opsiyonel).</i>"
        right_bgr = _limit_size(_load_bgr(right_img))
        left_bgr = _limit_size(_load_bgr(left_img))

        analyzer = _get_analyzer()
        multi = analyze_multi(front_bgr, analyzer, right_bgr, left_bgr)

        if "error" in multi:
            return None, "<b style='color:#e05'>Karşıdan fotoğrafta yüz bulunamadı — daha net, tek yüz içeren, tam karşıdan bir fotoğraf deneyin.</b>"

        canvas_bgr = draw_overlay(front_bgr, multi["front"], scores=multi["scores"], include_panel=False)
        canvas_rgb = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
        html = _score_table_html(multi)
        return canvas_rgb, html
    except Exception as e:
        traceback.print_exc()
        return None, f"<b style='color:#e05'>Hata: {e}</b>"


def _mark_status(img):
    if img is not None:
        return "<div class='step-status'><span class='sa-added'>✅ Eklendi</span></div>"
    return "<div class='step-status'><span class='sa-missing'><span class='dot-missing'></span>Henüz eklenmedi</span></div>"


CSS = """
.gradio-container{
  background: radial-gradient(circle at 50% -10%, #dff3ee 0%, #eef2f6 55%) !important;
  min-height: 100vh;
}
footer{ display:none !important; }
#phone{
  max-width: 440px; margin: 28px auto 48px auto !important;
  background:#ffffff; border-radius:34px;
  box-shadow: 0 25px 60px -18px rgba(15,45,40,.28), 0 0 0 1px rgba(15,45,40,.05);
  overflow:hidden; padding:0 !important;
}
#phone-header{
  background: linear-gradient(135deg,#18a999,#0c6f68);
  color:#fff; padding:22px 20px 30px 20px; margin-bottom:-16px !important;
}
#phone-header h1{ font-size:20px; margin:0 0 4px 0; font-weight:800; }
#phone-header p{ font-size:12.5px; margin:0; opacity:.9; line-height:1.45 }
#phone-body{ padding: 0 16px 18px 16px !important; }
.step-card{
  background:#f6f9f9 !important; border-radius:18px !important; padding:4px 4px 10px 4px !important;
  margin-bottom:14px !important; border:1px solid #e9eef0 !important;
}
.step-title{ font-size:13.5px; font-weight:700; color:#0c6f68; padding:10px 10px 2px 10px; }
.step-title .req{ color:#d84343; font-weight:600; font-size:11.5px; }
.step-title .opt{ color:#8a9a98; font-weight:600; font-size:11.5px; }
.step-status{ padding:2px 10px 6px 10px; font-size:12px; }
.sa-added{ color:#0c8a63; font-weight:600; }
.sa-missing{ color:#9aa5a3; }
#analyze-btn{
  border-radius:16px !important; height:50px !important; font-weight:800 !important;
  font-size:15.5px !important; margin-top:4px !important;
  box-shadow: 0 10px 24px -8px rgba(12,111,104,.55) !important;
}
#result-img{ border-radius:18px !important; overflow:hidden; }
.sa-panel{ font-family:inherit; color:#1d2b29; padding: 4px 2px; }
.sa-panel h4{ margin:16px 0 8px 0; font-size:14px; color:#0c6f68; }
.sa-panel h4:first-child{ margin-top:4px; }
.sa-qline{ font-size:13px; margin:5px 0; color:#33413f; }
.sa-row{ margin:12px 0; }
.sa-row-top{ display:flex; justify-content:space-between; font-size:13.5px; color:#1d2b29; }
.sa-dot{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }
.sa-track{ background:#e7edec; border-radius:7px; height:9px; overflow:hidden; margin-top:4px; }
.sa-fill{ height:100%; border-radius:7px; }
.sa-note{ font-size:11px; color:#8a9a98; margin-top:3px; }
.sa-hint{ font-size:11px; color:#8a9a98; margin-top:10px; line-height:1.5; }
.sa-hint-inline{ font-weight:400; font-size:11.5px; color:#8a9a98; }
.step-num{
  display:inline-flex; align-items:center; justify-content:center;
  width:19px; height:19px; border-radius:50%; background:#0c6f68; color:#fff;
  font-size:11.5px; font-weight:800; margin-right:6px; vertical-align:middle;
}
.dot-missing{
  display:inline-block; width:9px; height:9px; border-radius:50%;
  background:#c7d0ce; margin-right:6px; vertical-align:middle;
}
"""

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="emerald",
    radius_size="lg",
    font=["Segoe UI", "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Cilt Analizi", theme=THEME, css=CSS) as demo_app:
    with gr.Column(elem_id="phone"):
        with gr.Column(elem_id="phone-header"):
            gr.HTML(
                "<h1>🔬 Cilt Analizi</h1>"
                "<p>Klasik görüntü işleme (OpenCV) tabanlı prototip. Tamamen yerel çalışır, "
                "hiçbir fotoğraf internete gönderilmez.</p>"
            )
        with gr.Column(elem_id="phone-body"):
            _missing_html = "<div class='step-status'><span class='sa-missing'><span class='dot-missing'></span>Henüz eklenmedi</span></div>"

            with gr.Group(elem_classes=["step-card"]):
                gr.HTML("<div class='step-title'><span class='step-num'>1</span>Tam karşıdan <span class='req'>(zorunlu)</span></div>")
                front_in = gr.Image(sources=["upload", "webcam"], type="numpy", mirror_webcam=False,
                                     height=230, show_label=False, container=False)
                front_status = gr.HTML(_missing_html)

            with gr.Group(elem_classes=["step-card"]):
                gr.HTML("<div class='step-title'><span class='step-num'>2</span>Başı hafifçe sağa çevrik <span class='opt'>(opsiyonel, daha güvenilir skor)</span></div>")
                right_in = gr.Image(sources=["upload", "webcam"], type="numpy", mirror_webcam=False,
                                     height=230, show_label=False, container=False)
                right_status = gr.HTML(_missing_html)

            with gr.Group(elem_classes=["step-card"]):
                gr.HTML("<div class='step-title'><span class='step-num'>3</span>Başı hafifçe sola çevrik <span class='opt'>(opsiyonel, daha güvenilir skor)</span></div>")
                left_in = gr.Image(sources=["upload", "webcam"], type="numpy", mirror_webcam=False,
                                    height=230, show_label=False, container=False)
                left_status = gr.HTML(_missing_html)

            btn = gr.Button("✨ Analiz Et", variant="primary", elem_id="analyze-btn")

            img_out = gr.Image(label="Sonuç", show_label=False, elem_id="result-img")
            html_out = gr.HTML()

    front_in.change(fn=_mark_status, inputs=front_in, outputs=front_status)
    right_in.change(fn=_mark_status, inputs=right_in, outputs=right_status)
    left_in.change(fn=_mark_status, inputs=left_in, outputs=left_status)

    btn.click(fn=run_analysis, inputs=[front_in, right_in, left_in], outputs=[img_out, html_out])

if __name__ == "__main__":
    demo_app.launch(inbrowser=True)
