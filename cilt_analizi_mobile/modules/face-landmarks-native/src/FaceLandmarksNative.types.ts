export type FaceRegions = {
  faceContour: [number, number][];
  leftEyebrow: [number, number][];
  rightEyebrow: [number, number][];
  leftEye: [number, number][];
  rightEye: [number, number][];
  nose: [number, number][];
  noseCrest: [number, number][];
  medianLine: [number, number][];
  outerLips: [number, number][];
  innerLips: [number, number][];
};

export type FaceDetectionResult =
  | { found: false }
  | { found: true; imageWidth: number; imageHeight: number; regions: FaceRegions };
