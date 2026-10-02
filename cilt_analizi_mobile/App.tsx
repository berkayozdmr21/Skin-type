import { useState } from 'react';
import {
  Alert,
  Image,
  Pressable,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
// NOT: Gercek yuz tespiti (Apple Vision, native Swift modulu - bkz. faceLandmarks.ts
// ve modules/face-landmarks-native/) su an BAGLI DEGIL. Calismasi icin Expo Dev Client
// uzerinden EAS ile derleme gerekiyor, bu da Apple Developer Program uyeligi ($99/yil)
// gerektiriyor - bu asamada bu karar alinmadi. Kod ileride kullanilmak uzere duruyor.

type ViewKey = 'front' | 'right' | 'left';

type PhotoState = Record<ViewKey, string | null>;

const STEP_INFO: Record<ViewKey, { num: string; title: string; tag: string; required: boolean }> = {
  front: { num: '1', title: 'Tam karşıdan', tag: '(zorunlu)', required: true },
  right: { num: '2', title: 'Başı hafifçe sağa çevrik', tag: '(opsiyonel, daha güvenilir skor)', required: false },
  left: { num: '3', title: 'Başı hafifçe sola çevrik', tag: '(opsiyonel, daha güvenilir skor)', required: false },
};

async function pickFrom(source: 'camera' | 'library'): Promise<string | null> {
  const perm =
    source === 'camera'
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (!perm.granted) {
    Alert.alert('İzin gerekli', source === 'camera' ? 'Kamera izni verilmedi.' : 'Galeri izni verilmedi.');
    return null;
  }
  const result =
    source === 'camera'
      ? await ImagePicker.launchCameraAsync({ quality: 0.8, cameraType: ImagePicker.CameraType.front })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });
  if (result.canceled) return null;
  return result.assets[0].uri;
}

function PhotoStep({
  view,
  uri,
  onPick,
  onClear,
}: {
  view: ViewKey;
  uri: string | null;
  onPick: (uri: string) => void;
  onClear: () => void;
}) {
  const info = STEP_INFO[view];

  const choose = () => {
    Alert.alert('Fotoğraf ekle', undefined, [
      { text: 'Kamerayla çek', onPress: async () => { const u = await pickFrom('camera'); if (u) onPick(u); } },
      { text: 'Galeriden seç', onPress: async () => { const u = await pickFrom('library'); if (u) onPick(u); } },
      { text: 'Vazgeç', style: 'cancel' },
    ]);
  };

  return (
    <View style={styles.stepCard}>
      <View style={styles.stepTitleRow}>
        <View style={styles.stepNum}>
          <Text style={styles.stepNumText}>{info.num}</Text>
        </View>
        <Text style={styles.stepTitle}>{info.title}</Text>
        <Text style={info.required ? styles.reqTag : styles.optTag}> {info.tag}</Text>
      </View>

      <Pressable onPress={choose} style={styles.photoBox}>
        {uri ? (
          <Image source={{ uri }} style={styles.photoPreview} />
        ) : (
          <View style={styles.photoPlaceholder}>
            <Text style={styles.photoPlaceholderIcon}>📷</Text>
            <Text style={styles.photoPlaceholderText}>Dokun: kamera veya galeri</Text>
          </View>
        )}
      </Pressable>

      <View style={styles.stepStatusRow}>
        {uri ? (
          <Text style={styles.statusOk}>✅ Eklendi</Text>
        ) : (
          <Text style={styles.statusMissing}>⬤ Henüz eklenmedi</Text>
        )}
        {uri ? (
          <Pressable onPress={onClear}>
            <Text style={styles.clearLink}>Kaldır</Text>
          </Pressable>
        ) : null}
      </View>
    </View>
  );
}

export default function App() {
  const [photos, setPhotos] = useState<PhotoState>({ front: null, right: null, left: null });
  const [analyzing, setAnalyzing] = useState(false);

  const setPhoto = (view: ViewKey, uri: string | null) =>
    setPhotos((p) => ({ ...p, [view]: uri }));

  const onAnalyze = async () => {
    if (!photos.front) {
      Alert.alert('Eksik fotoğraf', 'Önce tam karşıdan bir fotoğraf ekleyin.');
      return;
    }
    setAnalyzing(true);
    // Fotoğraf akışı (kamera/galeri, 3 adım) çalışıyor. Gerçek yüz tespiti/skor motoru
    // henüz bağlı değil - bkz. App.tsx üstündeki not.
    setTimeout(() => {
      setAnalyzing(false);
      Alert.alert(
        'Analiz altyapısı henüz bağlı değil',
        'Fotoğraf akışı çalışıyor — yüz/landmark tespiti ve skor hesaplama daha sonra eklenecek.'
      );
    }, 600);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor="#0c6f68" />
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>🔬 Cilt Analizi</Text>
          <Text style={styles.headerSubtitle}>
            Klasik görüntü işleme tabanlı prototip. Tamamen yerel çalışır, hiçbir fotoğraf
            internete gönderilmez.
          </Text>
        </View>

        <View style={styles.body}>
          <PhotoStep view="front" uri={photos.front} onPick={(u) => setPhoto('front', u)} onClear={() => setPhoto('front', null)} />
          <PhotoStep view="right" uri={photos.right} onPick={(u) => setPhoto('right', u)} onClear={() => setPhoto('right', null)} />
          <PhotoStep view="left" uri={photos.left} onPick={(u) => setPhoto('left', u)} onClear={() => setPhoto('left', null)} />

          <Pressable style={styles.analyzeBtn} onPress={onAnalyze} disabled={analyzing}>
            <Text style={styles.analyzeBtnText}>{analyzing ? 'Analiz ediliyor…' : '✨ Analiz Et'}</Text>
          </Pressable>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const TEAL = '#0c6f68';

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#eef2f6' },
  scroll: { paddingBottom: 32 },
  header: {
    backgroundColor: TEAL,
    paddingTop: 18,
    paddingBottom: 26,
    paddingHorizontal: 20,
    borderBottomLeftRadius: 28,
    borderBottomRightRadius: 28,
  },
  headerTitle: { color: '#fff', fontSize: 22, fontWeight: '800', marginBottom: 6 },
  headerSubtitle: { color: '#fff', opacity: 0.9, fontSize: 13, lineHeight: 18 },
  body: { paddingHorizontal: 16, paddingTop: 16 },
  stepCard: {
    backgroundColor: '#f6f9f9',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#e9eef0',
    padding: 12,
    marginBottom: 14,
  },
  stepTitleRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' },
  stepNum: {
    width: 20, height: 20, borderRadius: 10, backgroundColor: TEAL,
    alignItems: 'center', justifyContent: 'center', marginRight: 7,
  },
  stepNumText: { color: '#fff', fontSize: 12, fontWeight: '800' },
  stepTitle: { fontSize: 14, fontWeight: '700', color: TEAL },
  reqTag: { fontSize: 11.5, color: '#d84343', fontWeight: '600' },
  optTag: { fontSize: 11.5, color: '#8a9a98', fontWeight: '600' },
  photoBox: {
    height: 180, borderRadius: 14, overflow: 'hidden', backgroundColor: '#e7edec',
  },
  photoPreview: { width: '100%', height: '100%', resizeMode: 'cover' },
  photoPlaceholder: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  photoPlaceholderIcon: { fontSize: 28, marginBottom: 6 },
  photoPlaceholderText: { color: TEAL, fontSize: 13, fontWeight: '600' },
  stepStatusRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    marginTop: 8, paddingHorizontal: 2,
  },
  statusOk: { color: '#0c8a63', fontWeight: '600', fontSize: 12.5 },
  statusMissing: { color: '#9aa5a3', fontSize: 12.5 },
  clearLink: { color: '#c0392b', fontSize: 12.5, fontWeight: '600' },
  analyzeBtn: {
    backgroundColor: TEAL, borderRadius: 16, height: 50,
    alignItems: 'center', justifyContent: 'center', marginTop: 4,
    shadowColor: TEAL, shadowOpacity: 0.4, shadowRadius: 12, shadowOffset: { width: 0, height: 8 },
    elevation: 4,
  },
  analyzeBtnText: { color: '#fff', fontSize: 15.5, fontWeight: '800' },
});
