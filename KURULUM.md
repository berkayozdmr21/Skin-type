# Kurulum — Kendi Bilgisayarınızda Çalıştırma

Önemli bir not: Ben (Claude) sizin bilgisayarınıza doğrudan erişemiyorum,
bir şey kuramıyorum. Bu yüzden size 2 adımda kendi başınıza
kurabileceğiniz, hazır bir paket bıraktım — aşağıdaki adımları izlemeniz
yeterli.

## Gereksinim

- Python 3.9 veya üzeri (yoksa: https://www.python.org/downloads/ —
  Windows'ta kurulum sırasında **"Add Python to PATH"** kutucuğunu
  işaretlemeyi unutmayın)

## Adım 1 — Kurulum

Bu klasördeki dosyaların olduğu yerde bir terminal/komut istemi açın
(Windows'ta klasöre girip adres çubuğuna `cmd` yazıp Enter'a basabilirsiniz)
ve şunu çalıştırın:

```bash
pip install -r requirements.txt
```

(Mac/Linux'ta `pip` çalışmazsa `pip3` deneyin.)

## Adım 2 — Çalıştırma

```bash
python app.py
```

(Mac/Linux'ta gerekirse `python3 app.py`.)

İlk çalıştırmada yüz landmark modeli (~54 MB) otomatik olarak indirilir —
bunun için bir kerelik internet bağlantısı gerekir, birkaç saniye sürer.
Ardından tarayıcınız otomatik açılır ve arayüzü görürsünüz
(açılmazsa terminalde yazan `http://127.0.0.1:7860` adresine elle gidin).

## Kullanım

1. Sol taraftan bir fotoğraf yükleyin **veya** "webcam" sekmesinden
   kameranızla anlık çekin (kamera, tarayıcı `localhost` üzerinden
   çalıştığı için izin isteyecektir — izin verin).
2. **"Analiz Et"** butonuna basın.
3. Sağ tarafta ROI çizimleri + skor paneli gömülü sonuç görseli, altında
   da okunaklı bir skor listesi ve kamera kalite kontrolü (aydınlatma/poz/
   pozisyon) çıkar.

Hiçbir fotoğraf internete gönderilmez — her şey kendi bilgisayarınızda,
yerel olarak çalışır (yalnızca ilk çalıştırmadaki tek seferlik model
indirmesi hariç).

## Kapatma

Terminalde `Ctrl + C` tuşlarına basmanız yeterli.

## Sorun giderme

- **"python: command not found"** → `python3` yazmayı deneyin, ya da
  Python'ı PATH'e ekleyerek yeniden kurun.
- **"Yüz bulunamadı" hatası** → Fotoğrafta tek, net görünen bir yüz
  olduğundan emin olun; çok karanlık/yan profil fotoğraflarda tespit
  başarısız olabilir (bu, önceki mesajdaki "kalite kontrolü" adımının
  tam da çözmeye çalıştığı sorun).
- **Kamera izni açılmıyor** → Tarayıcınızın adres çubuğundaki kilit/kamera
  simgesine tıklayıp siteye kamera izni verdiğinizden emin olun.

## Bu bir prototip, unutmayın

Skorlardaki eşik değerleri (kalibrasyon) tek bir test fotoğrafıyla elle
ayarlandı — kesin bir ölçüm aracı değil. Detaylar ve sınırlamalar için
`skin_analysis.py` içindeki yorum satırlarına ve önceki mesajdaki
fizibilite notlarına bakın.
