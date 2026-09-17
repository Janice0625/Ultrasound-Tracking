[README.md](https://github.com/user-attachments/files/32359118/README.md)
# Pediatric Spine Ultrasound Real-Time Tracking and Stitching System

A computer-assisted pediatric spinal ultrasound system for vertebral segmentation, real-time tracking, automatic labeling, image quality screening, and lumbar-sacral image stitching.

---

##  Project Overview

Pediatric spinal ultrasound is commonly used to evaluate spinal structures in newborns. However, ultrasound probes provide only a limited field of view, requiring clinicians to scan the lumbar and sacral regions separately and interpret vertebral positions manually.

This project develops an end-to-end pediatric spine ultrasound processing system that combines deep learning, computer vision, real-time tracking, and image stitching.

The system supports two major workflows:

- **Stitching Mode** for processing existing lumbar and sacral ultrasound images
- **Real-Time Mode** for detecting, labeling, counting, and tracking vertebrae from ultrasound video

The overall goal is to reduce the difficulty of identifying vertebral levels and provide a more intuitive visualization of the infant spine.

---

##  Key Features

### 1. U-Net Spine Segmentation

A U-Net segmentation model is used to separate spinal structures from the ultrasound background.

The segmentation result is further processed to extract vertebral contours and coordinate information for labeling and image stitching.

### 2. YOLO-Based Real-Time Vertebra Detection

YOLO is used to detect vertebral structures from real-time ultrasound video.

The system supports two operating modes:

#### Count Mode

- Detects vertebrae in the current frame
- Labels the rightmost detected vertebra as `1`
- Numbers the remaining vertebrae sequentially from right to left

#### Track Mode

- Tracks vertebral positions across consecutive frames
- Uses anatomical order and slope-based analysis to identify important vertebral landmarks
- Supports automatic localization of the L5 and S1 regions
- Generates labeled frames for further processing and stitching

### 3. Lumbar-Sacral Image Stitching

The system combines lumbar and sacral ultrasound images using vertebral coordinate information.

The stitching pipeline:

1. Preprocesses the ultrasound images
2. Segments spinal structures using U-Net
3. Detects and labels vertebral positions
4. Aligns the lumbar and sacral images using the L5 region
5. Applies weighted blending in the overlapping region
6. Generates a unified panoramic spine image

### 4. Automatic Error Screening

The system automatically checks whether an input image meets the expected structural requirements.

Examples include:

- Insufficient vertebral landmarks
- Missing L5 or S1 reference points
- Images that do not satisfy the predefined geometric conditions

When an invalid image is detected, the system can display warning messages such as:

```text
Wrong Picture
```

This mechanism helps filter unsuitable ultrasound images before further processing.

### 5. PyQt6 Graphical User Interface

A desktop GUI was developed using PyQt6 to integrate the complete workflow.

The interface provides:

- Ultrasound image input
- Image preprocessing
- U-Net segmentation
- Vertebral contour visualization
- Real-time video capture
- YOLO-based tracking
- Count Mode
- Track Mode
- Best-frame selection
- Lumbar-sacral image stitching
- Final result visualization

---

##  System Workflow

### Stitching Mode

```text
Ultrasound Images
       |
       v
Image Preprocessing
       |
       v
U-Net Segmentation
       |
       v
Vertebral Contour Detection
       |
       v
Coordinate Extraction
       |
       v
Lumbar-Sacral Image Alignment
       |
       v
Weighted Image Stitching
       |
       v
Panoramic Spine Image
```

Stitching Mode is designed for cases where lumbar and sacral ultrasound images have already been captured.

The system processes both images, identifies vertebral landmarks, aligns them around the L5 region, and produces a combined spinal image.

---

### Real-Time Mode

```text
Real-Time Ultrasound Video
          |
          v
       YOLO
          |
          v
   Select Operation Mode
       /         \
      /           \
 Count Mode     Track Mode
      |              |
      v              v
Sequential       Vertebral
Numbering        Tracking
                     |
                     v
              L5 / S1 Detection
                     |
                     v
               Best Frame
                 Selection
                     |
                     v
              Image Stitching
```

Real-Time Mode processes ultrasound video while the probe is being moved.

YOLO detects vertebral structures, after which the user can select either Count Mode or Track Mode.

Track Mode can further identify candidate L5 and S1 landmarks and generate images for final stitching.

---

##  Technologies

### Programming
- Python

### Deep Learning
- U-Net
- YOLOv10

### Computer Vision
- OpenCV
- Contour Detection
- Coordinate-Based Analysis
- Image Alignment
- Weighted Image Blending

### GUI
- PyQt6

### Development Environment
- Windows 11
- Python
- PyQt6

---

##  System Testing

The system was tested across multiple functional components, including:

- Image loading and preprocessing
- U-Net segmentation
- Vertebral contour labeling
- Automatic error detection
- Image stitching
- Real-time video recording
- Track Mode
- Count Mode
- Real-time stitching
- GUI interaction
- System integration

The test specification reports successful execution of the major system functions and stable operation under the test environment.

---

##  Example Processing Pipeline

```text
Input Ultrasound
      ↓
Preprocessing
      ↓
U-Net Segmentation
      ↓
Vertebral Detection / Contour Extraction
      ↓
YOLO Real-Time Tracking
      ↓
Count / Track Mode
      ↓
Landmark Detection
      ↓
Best Image Selection
      ↓
Lumbar-Sacral Stitching
      ↓
Final Spine Visualization
```

---

##  Project Goals

This project aims to:

- Assist in identifying lumbar and sacral vertebral structures
- Reduce the difficulty of manually counting vertebral levels
- Provide real-time vertebral tracking and visualization
- Automatically identify unsuitable ultrasound images
- Generate a more complete visualization of the infant spine through image stitching
- Provide an integrated and user-friendly workflow for spinal ultrasound image analysis

---

##  Team

**Chang Gung University**  
Department of Computer Science and Information Engineering

Project Members:

- Jou-Hui Yu
- Ting-Ting Teng
- Janice Lin

Advisors:

- Prof. Shih-Lin Wu
- Prof. Yueh-Peng Chen

---

##  Project Type

Senior Capstone Project  
Chang Gung University
