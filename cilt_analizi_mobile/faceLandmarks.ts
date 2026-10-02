import FaceLandmarksNative from './modules/face-landmarks-native/src/FaceLandmarksNativeModule';
import type { FaceDetectionResult, FaceRegions } from './modules/face-landmarks-native/src/FaceLandmarksNative.types';

export type { FaceDetectionResult, FaceRegions };

/** uri: expo-image-picker'dan gelen yerel dosya yolu (jpg). Apple Vision (native, iOS) ile calisir. */
export async function detectFace(uri: string): Promise<FaceDetectionResult> {
  return FaceLandmarksNative.detectFace(uri);
}
