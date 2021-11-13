# Automatic License Plate Recognition
- Basic image processing techniques (In controlled lighting conditions with predictable license plates).
- Dedicated object detectors (HOG + Linear SVM, Faster R-CNN, SSDs, and YOLO).
- State-of-the-art ANPR software utilizes Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs).

## OpenCV

## Optical Character Recognition (OCR)

### Page Segmentation Method (PSM)
- Setting that indicates layout analysis mode for the document / image.
- There are 13 modes of operation.
- The 7th: Treat the image as a single text line.

### Whitelist
- A listing of characters (i.e. letters, digits, symbols) that Tesseract will consider.

## Strategy
- Detect and localize a license plate in an input image.
- Extract the characters from the license plate.
- Apply OCR techniques to obtain the text.
