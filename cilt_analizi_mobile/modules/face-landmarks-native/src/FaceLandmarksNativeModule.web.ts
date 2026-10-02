import { registerWebModule, NativeModule } from 'expo';

class FaceLandmarksNativeModule extends NativeModule<{}> {}

export default registerWebModule(FaceLandmarksNativeModule, 'FaceLandmarksNativeModule');
