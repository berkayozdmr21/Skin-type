import { NativeModule, requireNativeModule } from 'expo';
import type { FaceDetectionResult } from './FaceLandmarksNative.types';

declare class FaceLandmarksNativeModule extends NativeModule<{}> {
  detectFace(uri: string): Promise<FaceDetectionResult>;
}

export default requireNativeModule<FaceLandmarksNativeModule>('FaceLandmarksNative');
