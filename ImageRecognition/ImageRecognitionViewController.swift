//
//  ImageRecognitionViewController.swift
//  ImageRecognition
//
//  Created by Yichi Zhang on 4/1/18.
//  Copyright © 2018 Yichi Zhang. All rights reserved.
//

import UIKit
import CoreML
import Vision   // 初步处理图片

// UIImagePickerControllerDelegate 选照片，UINavigationControllerDelegate 负责弹出
class ImageRecognitionViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {

    @IBOutlet weak var imageDisplay: UIImageView!
    @IBOutlet weak var display: UILabel!
    
    let imagePicker = UIImagePickerController()
    let libraryImagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        imagePicker.delegate = self
        imagePicker.sourceType = .camera
        imagePicker.allowsEditing = false
        
        libraryImagePicker.delegate = self
        libraryImagePicker.sourceType = .photoLibrary
        libraryImagePicker.allowsEditing = false
    }

    @IBAction func tablePhoto(_ sender: UIBarButtonItem) {
        
        present(imagePicker, animated: true, completion: nil)
    }
    
    @IBAction func selectPhoto(_ sender: UIBarButtonItem) {
        
        present(libraryImagePicker, animated: true, completion: nil)
    }
    
    // image picker delegate，picker(imagePicker, libraryImagePicker), info(image information)
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        //UIImagePickerControllerOriginalImage: Specifies the original, uncropped image selected by the user. The value for this key is a UIImage object.
        if let selectedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            imageDisplay.image = selectedImage
            
            // Core Image, A representation of an image to be processed or produced by Core Image filters.
            guard let coreImage = CIImage(image: selectedImage) else {
                print("cannot convert to Core Image")
                return
            }
            
            identifyImage(image: coreImage)
        }
        
        picker.dismiss(animated: true, completion: nil)
    }

    func identifyImage(image: CIImage) {
        
        // A container for a Core ML model used with Vision requests.
        do {
            let trainedModel = try VNCoreMLModel(for: Inceptionv3().model)
            
            // create request
            let coreMLRequest = VNCoreMLRequest(model: trainedModel, completionHandler: {(res, err) in
                guard let results = res.results as? [VNClassificationObservation] else {
                    print("Cannot get VNClassificationObservation")
                    return
                }
                
                // All possible results
                print(results)
                
                // Optimal result
                if let optimalResult = results.first {
                    let nf = NumberFormatter()
                    nf.numberStyle = NumberFormatter.Style.decimal
                    nf.maximumFractionDigits = 2
                    
                    self.display.text = "\(optimalResult.identifier) with confidence of \(nf.string(from: NSNumber(value: optimalResult.confidence * 100))!)%"
                }
            })
            
            // make request
            do {
                try VNImageRequestHandler(ciImage: image).perform([coreMLRequest])
            }
            catch {
                print("ML request failed: \(error)")
            }
        }
        catch {
            print("Load inception model failed \(error)")
        }
    }
}
