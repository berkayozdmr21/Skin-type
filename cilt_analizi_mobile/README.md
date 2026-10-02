# Cilt Analizi — Mobil (Expo / React Native)

Masaüstü uygulamanın mobil portu. 3 adımlı fotoğraf akışı (kamera/galeri: karşıdan,
sağa dönük, sola dönük) çalışır durumda.

## Durum

- Fotoğraf seçme/çekme arayüzü: çalışıyor, Expo Go ile test edildi.
- Gerçek yüz tespiti / cilt skoru hesaplama: **henüz bağlı değil.**

## Yüz tespiti için denenenler

- `@tensorflow/tfjs` + `tfjs-react-native`: React Native'in artık zorunlu olan
  "New Architecture"ı ile uyumsuz çıktı (runtime crash). Bu paketler kaldırıldı.
- `modules/face-landmarks-native/`: Apple'ın Vision framework'ünü (`VNDetectFaceLandmarksRequest`)
  kullanan native bir Expo modülü (Swift) — kod hazır, App.tsx'e henüz bağlanmadı.
  Çalışması için Expo Dev Client üzerinden EAS ile derleme gerekiyor, bu da gerçek
  iPhone'da test için Apple Developer Program üyeliği ($99/yıl) gerektiriyor.
  Bu adımda bu masrafa girilmedi, karar sonraya bırakıldı.

## Çalıştırma (Expo Go ile, foto akışını test etmek için)

```
npm install
npx expo start --lan
```

Telefon ve bilgisayar aynı Wi-Fi ağında olmalı. iOS'ta fiziksel cihazda Expo Go
kullanmak için hem Expo Go'da hem bilgisayarda (`npx expo login`) aynı Expo
hesabına giriş yapılmış olması gerekiyor (Apple'ın zorunlu kıldığı bir kural).
