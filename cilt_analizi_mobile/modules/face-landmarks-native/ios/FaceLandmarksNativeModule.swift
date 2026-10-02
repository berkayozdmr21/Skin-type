import ExpoModulesCore
import Vision
import UIKit

enum FaceLandmarksError: Error, LocalizedError {
  case invalidImage
  var errorDescription: String? {
    switch self {
    case .invalidImage: return "Goruntu okunamadi"
    }
  }
}

public class FaceLandmarksNativeModule: Module {
  public func definition() -> ModuleDefinition {
    Name("FaceLandmarksNative")

    // uri: yerel dosya yolu (expo-image-picker'dan gelen file:// URI).
    // Donus: { found, imageWidth, imageHeight, regions: { <ad>: [[x,y], ...] } }
    // regions piksel koordinatinda (sol-ust kaynak), goruntunun EXIF yonu duzeltilmis halde.
    AsyncFunction("detectFace") { (uri: String) -> [String: Any] in
      guard let url = URL(string: uri),
            let data = try? Data(contentsOf: url),
            let image = UIImage(data: data),
            let cgImage = image.cgImage else {
        throw FaceLandmarksError.invalidImage
      }

      let orientation = CGImagePropertyOrientation(image.imageOrientation)
      let handler = VNImageRequestHandler(cgImage: cgImage, orientation: orientation, options: [:])
      let request = VNDetectFaceLandmarksRequest()

      do {
        try handler.perform([request])
      } catch {
        return ["found": false]
      }

      guard let results = request.results,
            let face = results.first,
            let landmarks = face.landmarks else {
        return ["found": false]
      }

      // Vision, dondurulmus (orientation-corrected) goruntu boyutlarina gore calisir.
      // handler'a cgImage + orientation verdigimiz icin normalizedPoints zaten
      // "dik" goruntuye gore; cikis boyutunu da ona gore hesaplamamiz gerekiyor.
      let uprightSize = cgImage.uprightSize(for: orientation)
      let w = uprightSize.width
      let h = uprightSize.height
      let bb = face.boundingBox // normalized [0,1], sol-alt kaynakli

      func regionPoints(_ region: VNFaceLandmarkRegion2D?) -> [[Double]] {
        guard let region = region else { return [] }
        return region.normalizedPoints.map { p in
          let normX = bb.origin.x + Double(p.x) * bb.size.width
          let normYBottomLeft = bb.origin.y + Double(p.y) * bb.size.height
          let normYTopLeft = 1.0 - normYBottomLeft
          return [normX * w, normYTopLeft * h]
        }
      }

      let regions: [String: [[Double]]] = [
        "faceContour": regionPoints(landmarks.faceContour),
        "leftEyebrow": regionPoints(landmarks.leftEyebrow),
        "rightEyebrow": regionPoints(landmarks.rightEyebrow),
        "leftEye": regionPoints(landmarks.leftEye),
        "rightEye": regionPoints(landmarks.rightEye),
        "nose": regionPoints(landmarks.nose),
        "noseCrest": regionPoints(landmarks.noseCrest),
        "medianLine": regionPoints(landmarks.medianLine),
        "outerLips": regionPoints(landmarks.outerLips),
        "innerLips": regionPoints(landmarks.innerLips),
      ]

      return [
        "found": true,
        "imageWidth": w,
        "imageHeight": h,
        "regions": regions,
      ]
    }
  }
}

private extension CGImage {
  /// orientation duzeltmesi uygulandiktan sonraki (dik) genislik/yukseklik.
  func uprightSize(for orientation: CGImagePropertyOrientation) -> CGSize {
    switch orientation {
    case .left, .right, .leftMirrored, .rightMirrored:
      return CGSize(width: height, height: width)
    default:
      return CGSize(width: width, height: height)
    }
  }
}
